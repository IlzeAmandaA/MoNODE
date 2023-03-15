import numpy as np
from scipy.ndimage import rotate
from torchvision import datasets
import torch

class MNIST(object):
    def __init__(self, data_path, dtype) -> None:
        self.mnist = datasets.MNIST(data_path, train=True, download=True)
        self.dtype = dtype

    def _collate_fn(self, videos):
        """
        Collate function for the PyTorch data loader.

        Merges all batch videos in a tensor with shape (length, batch, channels, width, height) and converts their pixel
        values to [0, 1].

        Parameters
        ----------
        videos : list
            List of uint8 NumPy arrays representing videos with shape (length, batch, width, height, channels).

        Returns
        -------
        torch.*.Tensor
            Batch of videos with shape (batch,length,channels, width, height) and float values lying in [0, 1].
        """
        seq_len = len(videos[0])
        batch_size = len(videos)
        nc = 1 if videos[0].ndim == 3 else 3
        w = videos[0].shape[1]
        h = videos[0].shape[2]
        tensor = torch.zeros((batch_size,seq_len, nc, h, w), dtype = self.dtype)
        for i, video in enumerate(videos):
            if nc == 1:
               # tensor[:, i, 0] += torch.from_numpy(video)
                tensor[i,:,0]  += torch.from_numpy(video)
            if nc == 3:
                tensor[:, i] += torch.from_numpy(np.moveaxis(video, 3, 1))
        tensor = tensor.type(self.dtype) #tensor.float()
        tensor = tensor / 255
        return tensor

class RotatingMNIST(MNIST):
    def __init__(self, data_path, data_n, n_angles, digit, frame_size=28, dtype=torch.float64) -> None:
        super().__init__(data_path, dtype)
        self.digit = digit
        self.n_angles = n_angles
        self.data_n = data_n
        self.frame_size = frame_size

    def _sample_digit(self):
        self.data_digit_idx  = torch.where(self.mnist.targets == self.digit)
        data_digit_imgs = self.mnist.data[self.data_digit_idx]
        random_idx = np.random.randint(0, data_digit_imgs.shape[0], self.data_n)
        self.data_digit_imgs = data_digit_imgs[random_idx]

    def _gen_angles(self):
        angles = np.linspace(0, 2 * np.pi, self.n_angles)[1:]
        self.angles = np.rad2deg(angles)
    
    def _sample_rotation(self):
        """ Rotate the input MNIST image in angles specified """
        rotated_imgs = np.array(self.data_digit_imgs).reshape((-1, 1, self.frame_size, self.frame_size))
        print(self.data_digit_imgs.shape)
        for a in self.angles:
            rotated_imgs = np.concatenate(
                (
                    rotated_imgs,
                    rotate(self.data_digit_imgs, a, axes=(1, 2), reshape=False).reshape((-1, 1, self.frame_size, self.frame_size)),
                ),
                axis=1,
            )
        return rotated_imgs

    def _gen_angles_stochastic(self):
        import math
        x0 = torch.rand([self.data_n]) * 30 - 15
        mu = torch.zeros_like(x0) # (torch.arange(1,6)-2).repeat(20,1).reshape(-1) * 3

        theta = 1.0
        sig   = math.sqrt(0.1) 

        def f(x):
            return theta*(x-mu) # + 5*theta*(math.pi*(x-mu)).sin()

        def g(x):
            return sig * math.sqrt(2*theta) * torch.randn_like(x)

        X  = [x0]
        dt = 0.01
        L = 10
        for t in range(L*self.n_angles-1):
            x_next = X[t] - dt*f(X[t]) + math.sqrt(dt)*g(X[t])
            X.append(x_next)
        
        self.angles = torch.stack(X[::L]).numpy().T # N,T
        # import matplotlib.pyplot as plt
        # plt.plot(self.angles.T[:,:10],'*-')
        # plt.savefig('deneme.png')
        # plt.close()

    def _sample_rotation_sequentially(self):
        """ Rotate the input MNIST image in angles specified """
        # self.data_digit_imgs is [N,28,28]
        # self.angles is [N,T]
        self.data_digit_imgs = self.data_digit_imgs.numpy()
        rot_imgs = []
        for it,(img,angles) in enumerate(zip(self.data_digit_imgs,self.angles)):
            rot_imgs_ = []
            angles = angles / 2 / np.pi * 360
            # angles = angles - angles[0]
            for angle in angles:
                rot_img = rotate(img.reshape(1,28,28), angle, axes=(1, 2), reshape=False)[0]
                rot_imgs_.append(rot_img.copy())
            rot_imgs.append(np.stack(rot_imgs_))
        rot_imgs = np.stack(rot_imgs) # N,T,28,28
        return rot_imgs


class MovingMNIST(MNIST):
    def __init__(self, data_path, style, seq_len = 15, max_speed = 4, frame_size=64, num_digits=2, dtype=torch.float64) -> None:
        super().__init__(data_path, dtype)
        self.style = style
        self.max_speed = max_speed
        self.seq_len = seq_len
        self.frame_size = frame_size
        self.num_digits = num_digits
        self.eps = 1e-8
        self.deterministic = True

    def _sample_style_idx(self):
        self.data_digit_idx = []
        for style_idx in self.style:
            for idx,target in enumerate(self.mnist.targets):
                if style_idx==target and idx not in self.data_digit_idx:
                    self.data_digit_idx.append(idx)
                    break
        
    def _subsample_data(self):
        self.data_sub = [np.array(img, dtype=np.uint8) for img in self.mnist.data[self.data_digit_idx]]

    def _sample_sequence(self):
        # generate videos
        x = np.zeros((self.seq_len, self.frame_size, self.frame_size), dtype=np.float32)
        # Generate the trajectories of each digit independently
        for n in range(self.num_digits):
            img = self.data_sub[np.random.randint(len(self.data_sub))]  # Random digit
            trajectory = self._compute_trajectory(*img.shape)  # Generate digit trajectory
            for t in range(self.seq_len):
                sx, sy, _, _ = trajectory[t]
                # Adds the generated digit trajectory to the video
                x[t, sx:sx + img.shape[0], sy:sy + img.shape[1]] += img
        # In case of overlap, brings back video values to [0, 255]
        x[x > 255] = 255
        return x.astype(np.uint8)


    def _compute_trajectory(self, nx, ny, init_cond=None):
        """
        Create a trajectory.

        Parameters
        ----------
        nx : int
            Width of digit image.
        ny : int
            Height of digit image.
        init_cond : tuple
            Optional initial condition for the generated trajectory. It is a tuple of integers (posx, poxy, dx, dy)
            where posx and poxy are the initial coordinates, and dx and dy form the initial speed vector.

        Returns
        -------
        list
            List of tuples (posx, poxy, dx, dy) describing the evolution of the position and speed of the moving
            object. Positions refer to the lower left corner of the object.
        """
        x = []  # Trajectory
        x_max = self.frame_size - nx  # Maximum x coordinate allowed
        y_max = self.frame_size - ny  # Maximum y coordinate allowed
        # Process or create the initial position and speed
        if init_cond is None:
            sx = np.random.randint(0, x_max + 1)
            sy = np.random.randint(0, y_max + 1)
            dx = np.random.randint(-self.max_speed, self.max_speed + 1)
            dy = np.random.randint(-self.max_speed, self.max_speed + 1)
        else:
            sx, sy, dx, dy = init_cond
        # Create the trajectory
        for t in range(self.seq_len):
            # After the movement of a timestep is applied, update the position and speed to take into account
            # collisions with frame borders
            sx, sy, dx, dy = self._process_collision(sx, sy, dx, dy, x_min=0, x_max=x_max, y_min=0, y_max=y_max)
            # Add rounded position and speed to the trajectory
            x.append([int(round(sx)), int(round(sy)), dx, dy])
            # Keep computing the trajectory with exact positions
            sy += dy
            sx += dx
        return x

    def _process_collision(self, sx, sy, dx, dy, x_min, x_max, y_min, y_max):
        """
        Takes as input current object coordinate and speed that might be over the frame borders after the movement of
        the last timestep, and updates them to take into account the object collision with frame borders.

        Parameters
        ----------
        sx : float
            Current object x coordinate, prior to checking whether it collided with a frame border.
        sy : float
            Current object y coordinate, prior to checking whether it collided with a frame border.
        dx : int
            Current object x speed, prior to checking whether it collided with a frame border.
        dy : int
            Current object y speed, prior to checking whether it collided with a frame border.
        x_min : int
            Minimum x coordinate allowed.
        x_max : int
            Maximum x coordinate allowed.
        y_min : int
            Minimum y coordinate allowed.
        y_max : int
            Maximum y coordinate allowed.

        Returns
        -------
        tuple
            Tuples (posx, poxy, dx, dy) of the position and speed of the moving object after a time unit. Positions
            refer to the lower left corner of the object.
        """
        # Check collision on all four edges
        left_edge = (sx < x_min - self.eps)
        upper_edge = (sy < y_min - self.eps)
        right_edge = (sx > x_max + self.eps)
        bottom_edge = (sy > y_max + self.eps)
        # Continue processing as long as a collision is detected
        while (left_edge or right_edge or upper_edge or bottom_edge):
            # Retroactively compute the collision coordinates, using the current out-of-frame position and speed
            # These coordinates are stored in cx and cy
            if dx == 0:  # x is onstant
                cx, cy = (sx, y_min) if upper_edge else (sx, y_max)
            elif dy == 0:  # y is constant
                cx, cy = (x_min, sy) if left_edge else (x_max, sy)
            else:
                a = dy / dx
                b = sy - a * sx
                # Searches for the first intersection with frame borders
                if left_edge:
                    left_edge, n = self._get_intersection_x(a, b, x_min, (y_min, y_max))
                    if left_edge:
                        cx, cy = n
                if right_edge:
                    right_edge, n = self._get_intersection_x(a, b, x_max, (y_min, y_max))
                    if right_edge:
                        cx, cy = n
                if upper_edge:
                    upper_edge, n = self._get_intersection_y(a, b, y_min, (x_min, x_max))
                    if upper_edge:
                        cx, cy = n
                if bottom_edge:
                    bottom_edge, n = self._get_intersection_y(a, b, y_max, (x_min, x_max))
                    if bottom_edge:
                        cx, cy = n
            # Displacement coefficient to get new coordinates after the bounce, taking into account the time left
            # (after all previous displacements) in the timestep to move the object
            p = ((sx - cx) / dx) if (dx != 0) else ((sy - cy) / dy)
            # In the stochastic case, randomly choose a new speed vector
            if not self.deterministic:
                dx = np.random.randint(-self.max_speed, self.max_speed + 1)
                dy = np.random.randint(-self.max_speed, self.max_speed + 1)
            # Reverse speed vector elements depending on the detected collision
            if left_edge:
                dx = abs(dx)
            if right_edge:
                dx = -abs(dx)
            if upper_edge:
                dy = abs(dy)
            if bottom_edge:
                dy = -abs(dy)
            # Compute the remaining displacement to be done during the timestep after the bounce
            sx = cx + dx * p
            sy = cy + dy * p
            # Check again collisions
            left_edge = (sx < x_min - self.eps)
            upper_edge = (sy < y_min - self.eps)
            right_edge = (sx > x_max + self.eps)
            bottom_edge = (sy > y_max + self.eps)
        # Return updated speed and coordinates
        return sx, sy, dx, dy

    def _get_intersection_x(self, a, b, x_lim, by):
        """
        Computes the intersection point of trajectory with the upper or lower border of the frame.

        Parameters
        ----------
        a : float
            dy / dx.
        b : float
            sy - a * sx.
        x_lim : int
            x coordinate of the border of the frame to test the intersection with.
        by : tuple
            Tuple of integers representing the frame limits on the y coordinate.

        Returns
        -------
        bool
            Whether the intersection point lies within the frame limits.
        tuple
            Couple of float coordinates representing the intersection point.
        """
        y_inter = a * x_lim + b
        if (y_inter >= by[0] - self.eps) and (y_inter <= by[1] + self.eps):
            return True, (x_lim, y_inter)
        return False, (x_lim, y_inter)

    def _get_intersection_y(self, a, b, y_lim, bx):
        """
        Computes the intersection point of trajectory with the left or right border of the frame.

        Parameters
        ----------
        a : float
            dy / dx.
        b : float
            sy - a * sx.
        y_lim : int
            y coordinate of the border of the frame to test the intersection with.
        bx : tuple
            Tuple of integers representing the frame limits on the x coordinate.

        Returns
        -------
        bool
            Whether the intersection point lies within the frame limits.
        tuple
            Couple of float coordinates representing the intersection point.
        """
        x_inter = (y_lim - b) / a
        if (x_inter >= bx[0] - self.eps) and (x_inter <= bx[1] + self.eps):
            return True, (x_inter, y_lim)
        return False, (x_inter, y_lim)

