from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt
import numpy as np
import scipy
import matplotlib
from matplotlib import image
from scipy.spatial import cKDTree as KDTree
import math
import torch
import torch.nn as nn
from scipy import interpolate
from collections import OrderedDict
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import time

def RemapRange(value, low1, high1, low2, high2):
    return low2 + (value - low1) * (high2 - low2) / (high1 - low1)
def RecenterDataForwardWithShape(v, x, y):
    return RemapRange(v, 0, max(x, y), -1, 1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def MSELoss0(preds, gt):
  return nn.MSELoss()(preds[0], gt)

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    # print(type(y))
    # print(type(x))
    # print('y', y.shape)
    # print('x', x.shape)
    if grad_outputs is None:
        #print('in')
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def gradients_mse(model_outputs, coords, gt_gradients):
    # print('mo',model_outputs.shape)
    # print('co',coords.shape)
    # print('gt',gt_gradients.shape)
    # compute gradients on the model
    gradients = gradient(model_outputs, coords)
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
    return gradients_loss

def gradients_mse_with_coords(preds,gt):
    return gradients_mse(preds[0],preds[1], gt)


def normDiff(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > 0:
        dx /= dist
        dy /= dist
    return (dx, dy)


def avgDiff(a, b, v):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > 0:
        dx /= dist
        dy /= dist

    resultx = a[0] + dx * dist * v
    resulty = a[1] + dy * dist * v

    return (resultx, resulty)


def distDiff(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dist = math.sqrt(dx * dx + dy * dy)

    return dist


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=5):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class CrowdPathDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, trajectories, obstacles=None,
                 multiplier=100):  # coords, dictionary, coord2keyFN, startPos, endPos, graph): #x_map, y_map, mask = None):
        'Initialization'
        
        self.trajectories = trajectories
        self.obstacles = obstacles
        self.obstacle_multiplier = multiplier
        self.buffer = .1

        numItems = 0
        for _, trajectory in self.trajectories.items():
            # print('coord',coord)
            numItems += len(trajectory) - 1
        if obstacles is not None:
            for _, obstacle in self.obstacles.items():
                # print('coord',coord)
                numItems += (len(obstacle) - 1) * self.obstacle_multiplier * 2

        self.totalpoints = numItems
        print("Total Midpoints:", self.totalpoints)


    def __len__(self):
        'Denotes the total number of samples'
        return 1

    def GetDirectionAlongTrajectory(self):

        input = np.zeros((1, self.totalpoints, 2), dtype=np.float32)
        output = np.zeros((1, self.totalpoints, 2), dtype=np.float32)
        # print('before',input.shape)

        current_midpoint = 0
        for _, trajectory in self.trajectories.items():  # for every starting position
           
            for i in range(len(trajectory) - 1):
                interp = np.random.random(1)[0]
                pos = avgDiff(trajectory[i], trajectory[i + 1], interp)

                dir = normDiff(trajectory[i], trajectory[i + 1])
                input[0, current_midpoint, 0] = pos[0]
                input[0, current_midpoint, 1] = pos[1]
                output[0, current_midpoint, 0] = dir[0]
                output[0, current_midpoint, 1] = dir[1]
                current_midpoint += 1

        if self.obstacles is not None:
            for k in range(self.obstacle_multiplier):
                for _, obstacle in self.obstacles.items():
                    for i in range(len(obstacle) - 1):
                        interp = np.random.random(1)[0]
                        # extra_buff = .001
                        # Get buffer wrt line segment (try to prevent overlap of line normals)
                        diff = distDiff(obstacle[i], obstacle[i + 1])
                        newbuff = self.buffer / diff
                        interp_with_buffer = interp * (1 - 2 * (newbuff)) + newbuff

                        pos = avgDiff(obstacle[i], obstacle[i + 1], interp_with_buffer)
                        # print(trajectory[i])
                        # print(trajectory[i+1])
                        # print(pos)
                        dir = normDiff(obstacle[i], obstacle[i + 1])
                        normal = (-dir[1], dir[0])

                        # forward
                        input[0, current_midpoint, 0] = pos[0] + normal[0] * self.buffer
                        input[0, current_midpoint, 1] = pos[1] + normal[1] * self.buffer
                        output[0, current_midpoint, 0] = normal[0]
                        output[0, current_midpoint, 1] = normal[1]
                        current_midpoint += 1

                        # backward
                        input[0, current_midpoint, 0] = pos[0] - normal[0] * self.buffer
                        input[0, current_midpoint, 1] = pos[1] - normal[1] * self.buffer
                        output[0, current_midpoint, 0] = -normal[0]
                        output[0, current_midpoint, 1] = -normal[1]
                        current_midpoint += 1

        return input, output

    def __getitem__(self, index):
        'Generates one sample of data'
        # print('getting a thing')
        return self.GetDirectionAlongTrajectory()

def train(network, data_generator, loss_function, optimizer):
    network.train()  # updates any network layers that behave differently in training and execution
    avg_loss = 0
    num_batches = 0
    for i, (input_data, target_output) in enumerate(data_generator):
        optimizer.zero_grad()  # Gradients need to be reset each batch
        prediction = network(input_data)  # Forward pass: compute the output class given a image
        loss = loss_function(prediction,target_output)  # Compute the loss: difference between the output and correct result
        loss.backward()  # Backward pass: compute the gradients of the model with respect to the loss
        optimizer.step()
        avg_loss += loss.item()
        num_batches += 1

    return avg_loss / num_batches

def test(network, test_loader, loss_function):
    network.eval()  # updates any network layers that behave differently in training and execution
    test_loss = 0
    num_batches = 0
    for data, target in test_loader:
        output = network(data)
        zero = torch.tensor(np.zeros(target.shape, dtype=np.float), dtype=torch.float32)
        test_loss += loss_function(output, target).item()
        num_batches += 1
    test_loss /= num_batches
    # print('\nTest set: Avg. loss: {:.4f})\n'.format(test_loss))
    return test_loss

def __getitem__(self, index):
    'Generates one sample of data'
    # print('getting a thing')
    return self.GetDirectionAlongTrajectory()

class Path:
    def __init__(self, points, sampling_rate = 0.1):
        self.orig_points = points
        self.sampled_points = []
        for point in range(1, len(points)):
            dist = np.sqrt(np.sum((points[point] - points[point - 1]) ** 2))
            samples = dist // sampling_rate
            self.sampled_points += [points[point-1]]
            if (samples > 1):
                step = (points[point] - points[point-1]) / samples
                sample_coordinate = points[point - 1]
                self.sampled_points += [sample_coordinate + i * step for i in range(1, int(samples))]
        self.sampled_points += [points[-1]]
    def export(self, file_ref):
        for point in self.sampled_points:
            file_ref.write("{x},{y}\n".format(x=point[0], y = point[1]))


class FieldInspector(QtWidgets.QMainWindow):
    def __init__(self, low_bound, high_bound, buffer):
        super().__init__()
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.buffer = buffer
        self.aspect_ratio = abs((high_bound[0] - low_bound[0] + 2 * buffer) / (high_bound[1] - low_bound[1] + 2 * buffer))
        self.dpi = 100
        self.canvasH = 1000
        self.canvasW = int(self.canvasH * self.aspect_ratio)
        self.gradient_label = QtWidgets.QLabel()
        self.gradient_plot = FigureCanvasQTAgg(Figure(figsize=(self.canvasW / self.dpi, self.canvasH / self.dpi), dpi=100))
        self.network_canvas = QtGui.QPixmap(self.canvasW, self.canvasH)
        #self.gradient_canvas.fill()
        #self.network_canvas.fill()
        #self.gradient_label.setPixmap(self.gradient_canvas)
        #self.network_label.setPixmap(self.network_canvas)


        self.xline_samples = 32
        self.mapped_ranges = [(self.low_bound[0] - self.buffer, self.low_bound[1] - self.buffer), (self.high_bound[0] + self.buffer, self.high_bound[1] + self.buffer)]
        self.xline_samples_frame = QtWidgets.QFrame()
        self.xline_samples_layout = QtWidgets.QVBoxLayout(self.xline_samples_frame)
        self.xline_samples_layout.setSpacing(5)
        self.xline_samples_layout.addStretch(1)
        self.xline_samples_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.xline_samples_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.xline_samples_label = QtWidgets.QLabel("Num Horizontal Samples: ", self.xline_samples_frame)
        self.xline_samples_label.setAlignment(Qt.AlignCenter)
        self.xline_samples_edit = QtWidgets.QLineEdit(str(self.xline_samples), self.xline_samples_frame)
        self.xline_samples_edit.setAlignment(Qt.AlignCenter)
        self.xline_samples_button = QtWidgets.QPushButton("Ok", self.xline_samples_frame)
        self.xline_samples_button.clicked.connect(self.updateXSamples)
        self.xline_samples_layout.addWidget(self.xline_samples_label)
        self.xline_samples_layout.addWidget(self.xline_samples_edit)
        self.xline_samples_layout.addWidget(self.xline_samples_button)
        self.xline_samples_edit.setText(str(self.xline_samples))

        self.display_options_frame = QtWidgets.QFrame()
        self.display_options_layout = QtWidgets.QVBoxLayout(self.display_options_frame)
        self.display_options_layout.setSpacing(5)
        self.display_options_layout.addStretch(1)
        self.display_options_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.display_options_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.gradient_checkbox = QtWidgets.QCheckBox(" : Show Gradient Arrows", self.display_options_frame)
        self.display_options_layout.addWidget(self.gradient_checkbox)
        self.gradient_checkbox.setChecked(True)
        self.gradient_checkbox.clicked.connect(self.generate_network_field_event)

        self.path_options_frame = QtWidgets.QFrame()
        self.path_options_layout = QtWidgets.QVBoxLayout(self.path_options_frame)
        self.path_options_layout.setSpacing(5)
        self.path_options_layout.addStretch(1)
        self.path_options_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.path_options_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.path_mode_combobox = QtWidgets.QComboBox(self.display_options_frame)
        self.path_mode_combobox.addItem("Draw Crowd Paths")
        self.path_mode_combobox.addItem("Draw Obstacle Paths")
        self.path_mode_combobox.addItem("Place Drop Test")
        self.path_options_layout.addWidget(self.path_mode_combobox)


        self.layout = QtWidgets.QHBoxLayout()
        self.control_panel = QtWidgets.QGridLayout()
        self.control_panel.setAlignment(Qt.AlignTop)
        self.control_panel.addWidget(self.xline_samples_frame, 0, 0)
        self.control_panel.addWidget(self.display_options_frame, 1, 0)
        self.control_panel.addWidget(self.path_options_frame, 2, 0)
        self.layout.addLayout(self.control_panel)
        self.layout.addWidget(self.gradient_label)

        self.central_widget = QtWidgets.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        self.setMouseTracking(True)
        self.paths = []
        self.obstacles = []
        self.points = []
        self.lines = [[[0], [0]]]
        self.prev_focus = -1

        self.setup_siren()
        self.generate_network_field()
        self.setup_tool_bar()
    def generate_network_field_event(self, e):
        self.generate_network_field()
    def updateXSamples(self, e):
        saved_xlinesamples = self.xline_samples
        text = self.xline_samples_edit.text()
        try:
            self.xline_samples = int(self.xline_samples_edit.text())
            if (self.xline_samples < 5 or self.xline_samples > 200):
                raise Exception()
            self.generate_network_field()
        except:
            self.xline_samples = self.xline_samples
            self.xline_samples_edit.setText("Invalid Entry: " + text)

    def setup_siren(self):
        self.nn = Siren(in_features=2, hidden_features=32, hidden_layers=3, out_features=1, outermost_linear=True)


    def setup_tool_bar(self):
        menubar = self.menuBar()

        exportAct = QtWidgets.QAction("Export", self)
        exportAct.setShortcut("Ctrl+S")
        exportAct.setStatusTip("Save drawn paths to csv file")
        exportAct.triggered.connect(self.export_paths)

        trainAct = QtWidgets.QAction("Train", self)
        trainAct.setShortcut("Ctrl+T")
        trainAct.setStatusTip("Train network off drawn paths")
        trainAct.triggered.connect(self.train_paths_and_reload)

        filemenu = menubar.addMenu("File")
        loadmenu = filemenu.addMenu("Load")

        optionmenu = menubar.addMenu("Options")

        actionmenu = menubar.addMenu("Actions")

        filemenu.addAction(exportAct)
        actionmenu.addAction(trainAct)

    def train_paths_and_reload(self):
        if (len(self.paths) and len(self.obstacles)):
            return
        conv_paths = {}
        for path in range(len(self.paths)):
            conv_paths[path] = list(map(lambda point: (point[0], point[1]), self.paths[path].sampled_points[::5]))
            # for p in conv_paths[path]:
                # print('point',p)
        conv_obstacles = {}
        for obstacle in range(len(self.obstacles)):
            conv_obstacles[obstacle] = list(map(lambda point: (point[0], point[1]), self.obstacles[obstacle].sampled_points[::5]))
        num_epochs = 1000
        learning_rate = 1e-4
        loss_function = gradients_mse_with_coords
        train_data_set = CrowdPathDataset(conv_paths, conv_obstacles, multiplier=1)
        train_data_generator = torch.utils.data.DataLoader(train_data_set, batch_size=1, shuffle=True)
        optimizer = torch.optim.Adam(self.nn.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            avg_loss = train(self.nn, train_data_generator, loss_function, optimizer)
            print('{}/{} : {}'.format(epoch+1,num_epochs,avg_loss))

        self.generate_network_field()
        self.paths = []
        self.obstacles = []


    def export_paths(self):
        if (len(self.paths) == 0):
            return
        options = QtWidgets.QFileDialog.Options()
        saveName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Path Export File Location","","Text Files (*.csv)", options=options)
        with open(saveName, 'w') as ref:
            for path in self.paths:
                path.export(ref)
        print("Export", saveName)

    def generate_network_field(self):
        sample_res = (self.mapped_ranges[1][0] - self.mapped_ranges[0][0]) / self.xline_samples
        num_y_samples = int((self.mapped_ranges[1][1] - self.mapped_ranges[0][0]) / sample_res)
        
        RecenterForward = lambda x : RecenterDataForwardWithShape(x,self.xline_samples,num_y_samples)
        
        sample_coords_offset = np.array( [[ [RecenterForward(j+.5),RecenterForward(i+.5)] for i in range(num_y_samples) for j in range(self.xline_samples) ]], dtype=np.float32)
        #sample_coords_offset = [np.array(((i + 0.5) * sample_res, (j + 0.5) * sample_res)) for i in range(self.xline_samples) for j in range(num_y_samples)]
        sample_coords_nn_preped = torch.unsqueeze(torch.from_numpy(np.array(sample_coords_offset, dtype=np.float32)), 0)
        potential_samples = self.nn(sample_coords_nn_preped)
        ax = self.gradient_plot.figure.subplots(1, 1)
        self.ax = ax
        ax.set_title("")
        ax.set_xlim(self.mapped_ranges[0][0], self.mapped_ranges[1][0])
        ax.set_ylim(self.mapped_ranges[0][1], self.mapped_ranges[1][1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax.set_aspect(1)
        ax.set_position([0, 0, 1, 1])
        ax.imshow(-potential_samples[0].cpu().view(num_y_samples, self.xline_samples).detach().numpy(), extent = [self.mapped_ranges[0][0], self.mapped_ranges[1][0], self.mapped_ranges[0][1], self.mapped_ranges[1][1]], interpolation='none')

        gradient_potential_samples = gradient(*self.nn(potential_samples[1]))

        gradient_potential_samples = gradient_potential_samples[0].cpu().view(self.xline_samples, num_y_samples, 2).detach().numpy()
        #print(gradient_potential_samples.shape)
        if (self.gradient_checkbox.isChecked()):
            coord_x, coord_y = np.meshgrid(list(map(lambda v: RemapRange(v + 0.5, 0, self.xline_samples, self.mapped_ranges[0][0], self.mapped_ranges[1][0]), range(self.xline_samples))), list(map(lambda v: RemapRange(v + 0.5, 0, num_y_samples, self.mapped_ranges[0][1], self.mapped_ranges[1][1]), range(num_y_samples))))
            #coord_x, coord_y = np.meshgrid(list(map(lambda v: RecenterForward(v+0.5), range(self.xline_samples))), list(map(lambda v: RecenterForward(v + 0.5), range(num_y_samples))))
            vx = gradient_potential_samples[:,:,0]
            vy = gradient_potential_samples[:,:,1]
            ux = vx/np.sqrt(vx**2 + vy**2)
            uy = vy/np.sqrt(vx**2 + vy**2)
            #ax.quiver(coord_x, coord_y, ux, uy, color='red')
            ax.quiver(coord_x, coord_y, ux, -uy, color='red')
        self.gradient_plot.figure.savefig("temp_fig.png")
        self.gradient_canvas = QtGui.QPixmap("temp_fig.png")
        self.gradient_label.setPixmap(self.gradient_canvas)

    def mouseReleaseEvent(self, event):
    
    
        sample_res = (self.mapped_ranges[1][0] - self.mapped_ranges[0][0]) / self.xline_samples
        num_y_samples = int((self.mapped_ranges[1][1] - self.mapped_ranges[0][0]) / sample_res)
        
        RecenterForward = lambda x : RecenterDataForwardWithShape(x,self.xline_samples,num_y_samples)
    
        if (self.path_mode_combobox.currentIndex() == 2):
            corrected_pos = np.array((self.mapped_ranges[0][0] + (self.mapped_ranges[1][0] - self.mapped_ranges[0][0]) * (event.x() - self.gradient_label.x()) / self.gradient_canvas.width(), self.mapped_ranges[0][1] + (self.mapped_ranges[1][1] - self.mapped_ranges[0][1]) * (event.y() - self.gradient_label.y()) / self.gradient_canvas.height()))
            drop_path = [(int(RemapRange(corrected_pos[0], self.mapped_ranges[0][0], self.mapped_ranges[1][0], 0, self.gradient_canvas.width())), int(RemapRange(corrected_pos[1], self.mapped_ranges[0][1], self.mapped_ranges[1][1], 0, self.gradient_canvas.height())))]
            step = 0
            while (corrected_pos[0] < self.mapped_ranges[1][0] and corrected_pos[0] > self.mapped_ranges[0][0] and corrected_pos[1] < self.mapped_ranges[1][1] and corrected_pos[1] > self.mapped_ranges[0][1] and step < 10000):
                nn_prep_pos = torch.unsqueeze(torch.from_numpy(np.array([corrected_pos], dtype=np.float32)), 0)
                pos_potentials = self.nn(nn_prep_pos)
                gradients = gradient(*self.nn(pos_potentials[1]))
                gradient_vals = gradients[0].cpu().view(1, 1, 2).detach().numpy()
                vx = gradient_vals[0][0][0]
                vy = gradient_vals[0][0][1]
                ux = vx / np.sqrt(vx ** 2 + vy ** 2)
                uy = vy / np.sqrt(vx ** 2 + vy ** 2)
                corrected_pos[0] += ux * .3
                corrected_pos[1] += uy * .3
                drop_path += [(int(RemapRange(corrected_pos[0], self.mapped_ranges[0][0], self.mapped_ranges[1][0], 0, self.gradient_canvas.width())), int(RemapRange(corrected_pos[1], self.mapped_ranges[0][1], self.mapped_ranges[1][1], 0, self.gradient_canvas.height())))]
                step += 1
            for point in range(len(drop_path) - 1):
                painter = QtGui.QPainter(self.gradient_label.pixmap())
                pen = painter.pen()
                pen.setWidth(6)
                pen.setColor(QtGui.QColor("#FFFFFF"))
                painter.setPen(pen)
                painter.drawLine(drop_path[point][0], drop_path[point][1], drop_path[point + 1][0], drop_path[point + 1][1])
                painter.end()
            self.update()
            return

        if (not len(self.points)):
            return
        if (self.path_mode_combobox.currentIndex() == 0):
            painter = QtGui.QPainter(self.gradient_label.pixmap())
            pen = painter.pen()
            pen.setWidth(6)
            pen.setColor(QtGui.QColor("#FF00FF"))
            painter.setPen(pen)
            painter.drawPoint(self.points[-1][0] + self.mapped_ranges[0][0], self.points[-1][1] + self.mapped_ranges[0][1])

            pen = painter.pen()
            pen.setWidth(6)
            pen.setColor(QtGui.QColor("#FFFF00"))
            painter.setPen(pen)
            painter.drawPoint(self.points[0][0] + self.mapped_ranges[0][0], self.points[0][1] + self.mapped_ranges[0][1])
            painter.end()
            self.update()

        #self.network_converted_points = list(map(lambda point: np.array((RemapRange(point[0], 0, self.network_canvas.width(), self.mapped_ranges[0][0], self.mapped_ranges[1][0]), RemapRange(point[1], 0, self.network_canvas.height(), self.mapped_ranges[0][1], self.mapped_ranges[1][1]))), self.points))
        
        self.network_converted_points = list(map(lambda point: np.array((RecenterDataForwardWithShape(point[0], self.network_canvas.width(), self.network_canvas.height()), RecenterDataForwardWithShape(point[1], self.network_canvas.width(), self.network_canvas.height()))), self.points))
        
        #self.network_converted_points = list(map(lambda point: np.array((RecenterForward(point[0]), RecenterForward(point[1], ))), self.points))
        
        if (self.path_mode_combobox.currentIndex() == 0):
            self.paths += [Path(self.network_converted_points)]
        else:
            self.obstacles += [Path(self.network_converted_points)]
        self.points = []
        self.prev_focus = -1


    def mouseMoveEvent(self, event):

        corrected_pos = (event.x() - self.gradient_label.x(), event.y() - self.gradient_label.y())
        if not (corrected_pos[0] >= 0 and corrected_pos[0] <=  self.gradient_canvas.width() and corrected_pos[1] >= 0 and corrected_pos[1] <=  self.gradient_canvas.height()):
            return
        if (len(self.points)):
            painter = QtGui.QPainter(self.gradient_label.pixmap())
            pen = painter.pen()
            pen.setWidth(4)
            if (self.path_mode_combobox.currentIndex() == 0):
                pen.setColor(QtGui.QColor("#0000FF"))
            elif (self.path_mode_combobox.currentIndex() == 1):
                pen.setColor(QtGui.QColor("#000000"))
            elif(self.path_mode_combobox.currentIndex() == 2):
                return
            painter.setPen(pen)
            painter.drawLine(self.points[-1][0] + self.mapped_ranges[0][0], self.points[-1][1] + self.mapped_ranges[0][1], corrected_pos[0], corrected_pos[1])
            painter.end()
        self.points += [(corrected_pos[0] - self.mapped_ranges[0][0], corrected_pos[1] - self.mapped_ranges[0][1])]
        self.update()


class FieldInspectorApp(QtWidgets.QApplication):
    def __init__(self, low_bound, high_bound, buffer):
        super().__init__([])
        self.window = FieldInspector(low_bound, high_bound, buffer)
        self.window.show()
        self.exec_()

app = FieldInspectorApp((-3, -3), (3, 3), 1.0)