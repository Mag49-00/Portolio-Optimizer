import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation as anim
from matplotlib.animation import PillowWriter as pw
from PortfolioModel import PortfolioModel
from matplotlib.axes import Axes

class Population:
    '''
    Abstract class which represents a population of investment portfolios as well as an optimization algorithm for
    that population.

    Supports methods to update the data at each timestep, return the best option within the current population as
    defined by the portfolio_model's error function, and print a detailed report on what the optimization algorithm
    has settled on, and a comparison with the defined targets.

    Parameters
    ----------
    portfolio_model : PortfolioModel
        The portfolio model used for optimization.

    number_particles : int
        The number of individual portfolios to simulate.

    Attributes
    ----------
    particles : np.ndarray
        The portfolio matrix representing the population. Each row represents one portfolio in the population, while
        each column represents a specific investment. The number at row x, column y should be the allocation of
        investment y in portfolio x. Negative values and other non-physical investment portfolios are supported (e.g.
        portfolios whose sum of allocations is not 1)

        This exactly corresponds to the definition of "portfolio array" as outlined in the PortfolioModel class.

        By default, creates random particles uniformly within the annulus starting at distance 0.2 from the origin
        and ending at distance 1 from the origin.

    model : PortfolioModel
        The portfolio model used for optimization.

    number_particles : int
        The total number of investment portfolios in the population

    number_investments : int
        The total number of investment options in the model.
    '''
    def __init__(self, portfolio_model: PortfolioModel, number_particles: int = 1):
        vector_size = portfolio_model.names.shape[0]
        
        rng = np.random.default_rng()
        directions = rng.normal(size = (number_particles, vector_size))
        directions = directions/np.linalg.norm(directions, axis = 1, keepdims = True)
        radii = rng.uniform(0.2, 1, size = (number_particles, 1))
        self.particles = directions * radii
        
        self.model = portfolio_model
        self.number_particles, self.number_investments = self.particles.shape

    def move(self, velocity: np.ndarray) -> None:
        '''
        Updates self.particles by applying the given velocity vector to them.
        '''
        self.particles += velocity

    def normalized(self) -> np.ndarray:
        '''
        Returns a normalized version of self.particles - identical to PortfolioModel.normalize
        '''
        return PortfolioModel.normalize(self.particles)

    def update(self) -> None:
        '''
        Method used to update the portfolio population at each time step, to be defined by daughter classes.
        '''
        pass

    def get_best(self) -> tuple[float, np.ndarray]:
        '''
        Finds the portfolio with the lowest error among the population.

        Returns
        -------
        float
            The numerical error of the best portfolio in the population.

        np.ndarray
            The one dimensional row of self.particles corresponding to the portfolio with the lowest error in the
            population.
        '''
        errors = self.model.total_error(self.normalized())
        
        best_index = errors.argmin()
        best_error = errors[best_index]

        return(best_error, self.normalized()[best_index])
        

    def simulate(self, iterations: int = 100):
        '''
        Repeatedly calls self.update() to calculate multiple steps of the optimization model.
        
        Parameters
        ----------
        iterations : int
            The number of timesteps to simulate.
        '''
        for k in range(iterations):
            self.update()

    def report(self):
        '''
        Prints a report of the optimization model, including the best portfolio found, its error, allocations,
        statistics, and comparison to the target values.
        '''
        best_error, best_index = self.get_best()
        print("-=Portfolio=-")
        print(self.model.portfolio_string(best_index))
        print()
        print("Allocations:")
        print(self.model.portfolio_allocation_string(best_index))
        print()
        print('Desired Allocations:')
        print(self.model.desired_allocations_str())
        print()
        print("MPT Stats:")
        print(f'Alpha: {self.model.alpha(best_index)}, Beta: {self.model.beta(best_index)}')
        print()
        print(f'Desired Beta: {self.model.target_beta}')
        print()
        print(f'Error: {best_error}')
            
class GradDesc(Population):
    def __init__(self, portfolio_model: PortfolioModel, number_particles: int = 1, learn_rate: float = 1.0):
        '''
        Used to perform the Gradient Descent algorithm on a portfolio.

        The only updates to the Population class are a defined update function (which applies the Gradient Descent
        algorithm), and a draw method, which graphs the population of portfolios over time under this algorithm.

        Parameters
        ----------
        See Population for a description of portfolio_model and number_particles.

        learn_rate : float
            The learning speed used for the gradient descent algorithm.

        Attributes
        ----------
        learn_rate : float
            The learning speed used for the gradient descent algorithm.
        '''
        Population.__init__(self, portfolio_model, number_particles)
        self.learn_rate = learn_rate

    def update(self) -> None:
        '''
        Update the portfolio population at each time step, by moving it against the gradient at a rate of
        self.learn_rate.
        '''
        self.move(-self.learn_rate * self.model.partials(self.particles))

    def draw(self, filename: str, iterations: int = 1000, interval: int = 300, num_gridpoints: int = 20) -> None:
        '''
        Creates a gif image of the gradient descent process. The gif includes four figures:
        - The raw data portfolios moving with each timestep, with a heatmap representing the error function in the
        background.
        - The normalized portfolios moving with each timestep, with a normalized heatmap of the error function in
        the background.
        - A 3D plot of the raw data portfolios moving with each timestep across a wireframe surface representing
        the error function.
        - A 3D plot of the normalized portfolios moving across a wireframe surface representing the normalized error
        function.

        Note: This function only works when the number of investments considered is equal to two. If the number of
        investments were larger, the corresponding error function would live in a higher dimensional space, and hence
        could not be visualized in the same way on a two dimensional screen.

        Parameters
        ----------
        filename : str
            The filename to save the gif image to.

        iterations : int
            The number of timesteps to simulate before creating the gif.

        interval : int
            The number of milliseconds between frames.

        num_gridpoints : int
            The number of gridpoints on each axis of the wireframe.
        '''
        if self.number_investments != 2:
            raise ValueError('draw() function only supports populations with two investments.')
        fig = plt.figure(figsize = (12, 10))

        # Creates one figure for each plot to be included on the gif.
        wireplot_raw_ax = fig.add_subplot(221, projection = '3d', computed_zorder=False)
        wireplot_norm_ax = fig.add_subplot(222, projection = '3d', computed_zorder=False)
        heatmap_raw_ax = fig.add_subplot(223)
        heatmap_norm_ax = fig.add_subplot(224)
        
        raw_bounds = (-1, 1)
        norm_bounds = (0, 1)
        raw_grid = np.linspace(-1, 1, num_gridpoints)
        raw_grid = np.meshgrid(raw_grid, raw_grid)
        norm_grid = np.linspace(0, 1, num_gridpoints)
        norm_grid = np.meshgrid(norm_grid, norm_grid)

        X, Y = raw_grid
        X_norm, Y_norm = norm_grid

        gridpoints = np.column_stack((X.ravel(), Y.ravel()))
        normalized_points = np.column_stack((X_norm.ravel(), Y_norm.ravel()))
        
        errors = self.model.total_error(self.model.normalize(gridpoints)).reshape(X.shape)
        normalized_errors = self.model.total_error(normalized_points).reshape(X_norm.shape)

        def set_format(axis: Axes, raw: bool, wireplot: bool, title: str):
            '''
            A small helper function to handle a lot of the setup for each individual figure, such as creating the
            title, plotting the wireframe/heatmap, creating axis labels, etc.

            After plotting, prints (does not return) the report of the finished portfolio.

            Parameters
            ----------
            axis: Axes
                The axis to format.

            raw: bool
                True if the axis represents raw data, False if it should be normalized.

            wireplot: bool
                True if the axis represents a wireplot, False if it should be a heatmap.

            title: str
                The title to give to the plot.
            '''
            bounds = norm_bounds
            if raw:
                bounds = raw_bounds
                
            axis.set_ylim(bounds)
            axis.set_xlim(bounds)
            
            s = ''
            if raw:
                s = 'raw '
            elif not wireplot:
                axis.plot([0, 1], [1, 0], c = 'white', linestyle = '--', zorder = 1)
            else:
                x = np.linspace(0, 1, num_gridpoints)
                y = 1 - x
                validity_line = np.column_stack((x, y))
                z = self.model.total_error(validity_line)
                axis.plot(x, y, z, color = 'black', zorder = 1)
            axis.set_xlabel(f'{self.model.names[0]} {s}portfolio weight')
            axis.set_ylabel(f'{self.model.names[1]} {s}portfolio weight')

            axis.set_title(title)

            if wireplot:
                axis.set_zlabel('Error')
                if raw:
                    axis.plot_wireframe(X, Y, errors, zorder = -1)
                else:
                    axis.plot_wireframe(X_norm, Y_norm, normalized_errors, zorder = -1)
            else:
                err = normalized_errors
                if raw:
                    err = errors
                axis.imshow(err, extent = [bounds[0], bounds[1], bounds[0], bounds[1]], cmap = 'viridis')
                
        set_format(wireplot_raw_ax, True, True, 'Raw Data Wireplot')
        set_format(wireplot_norm_ax, False, True, 'Normalized Data Wireplot')
        set_format(heatmap_raw_ax, True, False, 'Raw Data Heatmap')
        set_format(heatmap_norm_ax, False, False, 'Normalized Data Heatmap')

        errors = self.model.total_error(self.normalized())
        normalized = self.normalized()
        
        raw_wireplot_scatter = wireplot_raw_ax.scatter(self.particles[:, 0], self.particles[:, 1], errors, c = 'red', zorder = 2)
        norm_wireplot_scatter = wireplot_norm_ax.scatter(normalized[:, 0], normalized[:, 1], errors, c = 'red', zorder = 2)
        raw_heat_scatter = heatmap_raw_ax.scatter(self.particles[:, 0], self.particles[:, 1], c = 'red', zorder = 2)
        norm_heat_scatter = heatmap_norm_ax.scatter(normalized[:, 0], normalized[:, 1], c = 'red', zorder = 2)

        def sketch(i: int):
            '''
            Another small helper function. This is just the function used by anim to create the gif.
            It updates the portfolio state, and then draws the updated particles to the gif.
            '''
            self.update()
            
            errors = self.model.total_error(self.normalized())
            normalized = self.normalized()
            
            raw_wireplot_scatter._offsets3d = (self.particles[:, 0], self.particles[:, 1], errors)
            norm_wireplot_scatter._offsets3d = (normalized[:, 0], normalized[:, 1], errors)
            raw_heat_scatter.set_offsets(self.particles)
            norm_heat_scatter.set_offsets(normalized)

        animation = anim(fig, sketch, interval = interval, frames = iterations)
        writer = pw(fps = 30)

        animation.save(filename, writer = writer)
        plt.close()
        self.report()

class ParticleSwarm(Population):
    '''
    Simulates a population under Particle Swarm optimization.

    Similar to GradDesc, includes a draw method to visualize the data. Also adds the relevant parameters to the
    underlying Population space that are needed by the algorithm, including inertia, cognitive, social, and random
    weights.

    Parameters
    ----------
    See Population for a description of portfolio_model and number_particles.

    inertia : float
        At each timestep, how much of each particle's current velocity should be carried over into the next timestep,
        before applying other forces.

    cognitive_coeff : float
        The weight each particle should give to moving towards its best historic location each timestep.

    social_coeff : float
        The weight each particle should give to moving towards the population's overall best historic location each
        timestep.

    rand_coeff : float
        The weight each particle should give to moving in a random direction each timestep.

    Attributes
    ----------
    particles : np.ndarray
        Works just like its parent class Population, however, since Particle Swarm does not share Gradient Descent's
        problems with the feasibility space for portfolios, it immediately creates particles only within that space
        (i.e., values are strictly nonnegative and sum up to one within a particular portfolio)

    velocity : np.ndarray
        Each row of this array represents the velocity corresponding to the portfolio of that same row in the
        population array.

    best_location : np.ndarray
        The portfolio corresponding to best_error; the portfolio that is, so far, the best found among all
        portfolios in the population.

    best_error : float
        The lowest error found by the population so far.

    best_locations_by_particle : np.ndarray
        Each row corresponds to the best location found by that particle in the population, i.e. for that row,
        the best portfolio that individual particle has found.

    best_error_by_particle : np.ndarray
        The errors corresponding to best_locations_by_particle, i.e. the errors at the best location found for each
        individual particle.

    inertia, cognitive_coeff, social_coeff, and rand_coeff all are defined in Parameters.
    
    '''
    def __init__(self, portfolio_model: PortfolioModel, number_particles: int = 1, inertia: float = 0.25, cognitive_coeff: float = 0.5, social_coeff: float = 0.5, rand_coeff: float = 1.0):
        Population.__init__(self, portfolio_model, number_particles)
        self.particles = np.abs(self.particles)
        self.particles /= self.particles.sum(axis = 1, keepdims = True)
        self.velocity = np.random.uniform(-1, 1, (number_particles, self.number_investments))
        self.best_location = None
        self.best_error = float('inf')
        self.best_locations_by_particle = np.empty_like(self.particles)
        self.best_error_by_particle = np.full(number_particles, np.inf)
        self.inertia = inertia
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.rand_coeff = rand_coeff

        self.update_bests()
        
    def update(self) -> None:
        '''
        Updates each particle in the population based on these rules:
        1. Move in the direction of the best location found by the population so far
        2. Move in the direction of the best location found by that particle so far
        3. Move in the same direction as the particle moved last step
        4. Move in a random direction

        The weights of each of these rules is determined by:
        - The global weight of each (inertia, cognitive_coeff, social_coeff, and rand_coeff)
        - A randomly generated number between 0 and 1 creating an additional multiplicative weight for this step
        only (no additional random weight is used for the random vector, as it is already random)

        After evaluating the new position, re-orients it back into feasibility space.
        '''
        cognitive_weights = np.random.uniform(0, 1, self.best_locations_by_particle.shape)
        social_weights = np.random.uniform(0, 1, self.best_locations_by_particle.shape)
        
        random_velocity = np.random.uniform(-1, 1, (self.number_particles, self.number_investments))

        self.velocity *= self.inertia
        self.velocity += (self.cognitive_coeff * cognitive_weights * (self.best_locations_by_particle - self.particles) + self.social_coeff * social_weights * (self.best_location - self.particles) + random_velocity * self.rand_coeff)
                                  
        self.move(self.velocity)

        self.particles = np.abs(self.particles)
        S = self.particles.sum(axis = 1, keepdims = True)
        self.particles = self.particles/S
        
        self.update_bests()

    def update_bests(self) -> None:
        '''
        Calculates and updates the new best locations and errors, both population-wide, and per-particle.
        '''
        errors = self.model.total_error(self.particles)
        
        best_index = errors.argmin()
        best_error = errors[best_index]
        
        if best_error < self.best_error:
            self.best_error = best_error
            self.best_location = self.particles[best_index].copy()

        improved = errors < self.best_error_by_particle
        
        self.best_error_by_particle[improved] = errors[improved]
        self.best_locations_by_particle[improved] = self.particles[improved]

    def get_best(self) -> tuple[float, np.ndarray]:
        '''
        Identical to the Population method it overrides, but now reports the best error/portfolio pair found so far,
        rather than the best among the current population. Since ParticleSwarm already stores this, the retrieval
        is much faster than computing the best error/portfolio pair again.
        '''
        return (self.best_error, self.best_location)


    def draw(self, filename, iterations: int = 1000, interval: int = 300, num_gridpoints: int = 20) -> None:
        '''
        Creates a gif image of the particle swarm process. The gif includes two figures:
        - The portfolios moving with each timestep, with a heatmap representing the error function in the background.
        - A 3D plot of the portfolios moving with each timestep across a wireframe surface representing the error
        function.

        Note: This function only works when the number of investments considered is equal to two. If the number of
        investments were larger, the corresponding error function would live in a higher dimensional space, and hence
        could not be visualized in the same way on a two dimensional screen.

        Parameters
        ----------
        filename : str
            The filename to save the gif image to.

        iterations : int
            The number of timesteps to simulate before creating the gif.

        interval : int
            The number of milliseconds between frames.

        num_gridpoints : int
            The number of gridpoints on each axis of the wireframe.
        '''
        
        if self.number_investments != 2:
            raise ValueError('draw() function only supports populations with two investments.')
        fig = plt.figure(figsize = (16, 6))
        wireplot_ax = fig.add_subplot(121, projection = '3d', computed_zorder=False)
        heatmap_ax = fig.add_subplot(122)
        
        bounds = (0, 1)
        grid = np.linspace(0, 1, num_gridpoints)
        grid = np.meshgrid(grid, grid)

        X, Y = grid

        gridpoints = np.column_stack((X.ravel(), Y.ravel()))
        
        errors = self.model.total_error(gridpoints).reshape(X.shape)

        wireplot_ax.clear()
        heatmap_ax.clear()
                
        wireplot_ax.set_ylim(bounds)
        wireplot_ax.set_xlim(bounds)
        heatmap_ax.set_ylim(bounds)
        heatmap_ax.set_xlim(bounds)
        
        wireplot_ax.set_xlabel(self.model.names[0] + ' portfolio weight')
        wireplot_ax.set_ylabel(self.model.names[1] + ' portfolio weight')
        heatmap_ax.set_xlabel(self.model.names[0] + ' portfolio weight')
        heatmap_ax.set_ylabel(self.model.names[1] + ' portfolio weight')

        wireplot_ax.set_title('Particle Swarm Wireplot')
        heatmap_ax.set_title('Particle Swarm Heatmap')

        wireplot_ax.set_zlabel('Error')
        wireplot_ax.plot_wireframe(X, Y, errors, zorder = -1)
        heatmap_ax.imshow(errors, extent = [bounds[0], bounds[1], bounds[0], bounds[1]], cmap = 'viridis')
        
        heatmap_ax.plot([0, 1], [1, 0], c = 'white', linestyle = '--', zorder = 1)

        x = np.linspace(0, 1, num_gridpoints)
        y = 1 - x
        validity_line = np.column_stack((x, y))
        z = self.model.total_error(validity_line)
        wireplot_ax.plot(x, y, z, color = 'black', zorder = 1)

        errors = self.model.total_error(self.particles)
        
        wireplot_scatter = wireplot_ax.scatter(self.particles[:, 0], self.particles[:, 1], errors, c = 'red', zorder = 2)
        heat_scatter = heatmap_ax.scatter(self.particles[:, 0], self.particles[:, 1], c = 'red', zorder = 2)

        def sketch(i):
            '''
            A small helper function. This is just the function used by anim to create the gif.
            It updates the portfolio state, and then draws the updated particles to the gif.
            '''
            self.update()
            
            errors = self.model.total_error(self.particles)
            
            wireplot_scatter._offsets3d = (self.particles[:, 0], self.particles[:, 1], errors)
            heat_scatter.set_offsets(self.particles)

        animation = anim(fig, sketch, interval = interval, frames = iterations)
        writer = pw(fps = 30)

        animation.save(filename, writer = writer)
        plt.close()
        self.report()
