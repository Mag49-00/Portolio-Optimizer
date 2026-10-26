import numpy as np
from math import e

class PortfolioModel:
    '''
    Represents the portfolio model used by optimization algorithms.

    Stores the investment characteristics for all investment options under consideration, as well as target
    parameters. Provides methods to calculate a portfolio's alpha, beta, and category allocations based on
    the individual investment parameters, and to calculate portfolio error based on the target parameters.
    Also provides a normalization method to translate raw vectors into viable portfolio allocations, and
    gradient calculations used for optimization algorithms such as Gradient Descent.

    Parameters
    ----------
    file : str
        The file to read parameters from when creating the model.
        
        The first line should be a comma-separated list of four values representing allocation_weight,
        alpha_weight, beta weight, and target_beta, respectively (as defined below).

        The second line should indicate the desired allocations for the model, with the format:
        Allocation 1 Type: allocation_1_value, Allocation 2 Type: allocation_2_value, ...
        Additional Notes on the Second line:
            - The allocations here need not include investment types which have a desired weight of zero.
            - It is best if the expected values sum to 1. If not, the error function will try to optimize
            towards an unattainable fund allocation, which may throw off the model.
            - Each fund is assumed to have exactly one allocation category, so if "Large Growth" and "Small
            Growth" are both included, "Growth" as its own standalone category should not be included.

        All subsequent lines define the individual investments, one per line. They should be of the form:
        Investment Name, investment_alpha, investment_beta, Investment Category

        Empty lines and extra spaces are ignored.
        Any line starting with '#' will be ignored as a comment.

        Example file format:
            #allocation weight, alpha weight, beta weight, target beta
            1, 0.3, 1, 0.8

            #desired allocations
            Large Value: 0.4, Large Growth: 0.6
            #Large Value: 0.2, Large Growth: 0.4, Small Growth: 0.1, Mid Blend: 0.1, Bond: 0.2

            #investments to consider (name, alpha, beta, category)
            ABC, -0.35, 0.6, Large Value
            DEF, 0.15, 1.2, Large Growth
            GHI, -1.2, 0.9, Small Growth
            JKL, -2.1, 0.8, Small Growth
            MNO, -5.4, 0.7, Mid Blend
            PQR, -6.8, 0.2, Bond
            STU, 1.5, 1.6, Crypto

    Attributes
    ----------
    alpha_weight : float | int
        The weight the model should assign to achieving high alpha.

    beta_weight : float | int
        The weight the model should assign to reaching its desired beta.

    allocation_weight : float | int
        The weight the model should assign to reaching the desired allocations.

    category_list : np.ndarray
        A one dimensional array of names (strings) of all possible allocation categories. The indices of
        this array and target_alloc_by_category correspond to the same allocation category.

    target_alloc_by_category : np.ndarray
        A one dimensional array with the desired percentage allocation in each allocation category.

    target_beta : float
        The desired value of beta the optimizer will look to achieve, as a 1 dimensional arrya.

    names : np.ndarray
        A one dimensional array containing the names of all the investment options as strings. The indices
        of this array, alphas, betas, and category_indices all correspond to the same investment option.

    alphas : np.ndarray
        A one dimensional array containing the alpha value of each individual investment option as floats

    betas : np.ndarray
        A one dimensional array containing the beta value of each individual investment option as floats.

    category_indices : np.ndarray
        A one dimensional array containing the integer index of each investment option's allocation category
        in category_list and target_alloc_by_category.

    category_matrix : np.ndarray
        A two dimensional array, where each row represents an investment option, and each column represents
        an investment category. The row indices correspond to the same indices in names, alphas, betas, and
        category_indices, while the column indices correspond to the same indices in category_list and
        target_alloc_by_category. The value at row x, column y is a one if the investment corresponding to
        index x is a member of the category corresponding to index y, and zero otherwise.

    Additional Note
    ---------------
    The methods normalize, alpha, beta, allocations_by_category, alpha_error, beta_error, allocation_error,
    total_error, and partials all take an np.ndarray as input corresponding to a portfolio array. For most
    of these, portfolio array may be one dimensional or two dimensional, but they are intended to be used
    primarily for use with two dimensional arrays. The functions normalize and partials explicitly *must*
    have two dimensional arrays as inputs.

    If one dimensional, the portfolio array should have length equal to the number of investment options,
    and represents a single portfolio composed of the available investment options. The value at index i
    should be the allocation of the portfolio within that particular investment (with indices aligned to
    correspond with the stored investment option arrays, e.g. names, alphas, etc.)
    
    if two dimensional, the portfolio array should have a number of columns equal to the number of investment
    options, again aligned to correspond with the stored investment arrays. Each row then represents an
    individual portfolio. The array may have any number of rows. This represents multiple portfolios
    simultaneously, one per row.

    The outputs of these methods also depends on the portfolio array dimensionality.
    - alpha/beta/alpha_error/beta_error/allocation_error/total_error: Collapses the dimension by 1. Each
    portfolio gets one value for each of these, so returns a number of floats equal to the number of rows in the
    original matrix (or one single float if the original matrix was one dimensional)

    - allocations_by_category: returns an array with the same number of rows as the original, but the number
    of columns changed to the number of investment categories rather than the total number of individual
    investments.

    - partials/normalize: returns an array of the same size as the original array.

    See Population.particles, as this exactly corresponds to portfolio arrays as defined here.
    '''

    def __init__(self, file: str):
        self.alpha_weight = 1
        self.beta_weight = 1
        self.allocation_weight = 1

        self.category_list = []
        self.target_alloc_by_category = []
        self.target_beta = 0.5

        self.names = []
        self.alphas = []
        self.betas = []

        self.category_indices = []
        self.category_matrix = []

        
        def reformat(x: str) -> str:
            '''
            Small helper function that takes a string as input, and removes spaces and line breaks from the
            string, and returns the result.
            '''
            while len(x)>0:
                if x[0] in ['\n', " "]:
                    x = x[1:]
                else:
                    break
            while len(x)>0:
                if x[-1] in ['\n', ' ']:
                    x = x[:-1]
                else:
                    break
            return x
        
        with open(file) as f:
            i = 0
            for line in f:
                line = reformat(line)
                if line != "":
                    if line[0]!="#":
                        items = [reformat(x) for x in line.split(',')]
                        if i == 0:
                            # Handling the first line, corresponding to the allocation_weight, alpha_weight, beta_weight, and target_beta.
                            self.allocation_weight, self.alpha_weight, self.beta_weight, self.target_beta = map(float, items)
                            i+=1
                        elif i == 1:
                            # Handling the second line, corresponding to the list of allocation categories.
                            for item in items:
                                category, amount = item.split(':')
                                self.category_list.append(category)
                                self.target_alloc_by_category.append(float(amount))
                            i+=1
                        else:
                            # Handling all other lines, which correspond to the definition of the parameters of a particular investment option.
                            name, alpha, beta, category = items
                            self.names.append(name)
                            self.alphas.append(float(alpha))
                            self.betas.append(float(beta))
                            if category not in self.category_list:
                                self.category_list.append(category)
                                self.target_alloc_by_category.append(0)
                                for j in range(len(self.category_matrix)):
                                    self.category_matrix[j].append(0)
                            self.category_matrix.append([1 if category == c else 0 for c in self.category_list])
                            self.category_indices.append(self.category_list.index(category))

        # After all data is loaded, cast the relevant lists to numpy arrays.
        self.category_list = np.array(self.category_list)
        self.target_alloc_by_category = np.array(self.target_alloc_by_category)
        self.names = np.array(self.names)
        self.alphas = np.array(self.alphas)
        self.betas = np.array(self.betas)
        self.category_indices = np.array(self.category_indices)
        self.category_matrix = np.array(self.category_matrix)

    @staticmethod
    def normalize(array: np.ndarray) -> np.ndarray:
        '''
        Takes a two dimensional array as input, and normalizes each row to the L1 norm using the transformation:
        xi => xi^2 / |x|^2. Then returns the normalized array.

        Given a portfolio array, this ensures that every entry in the output array is nonnegative, and the sum
        of all entries is 1, which guarantees the result represents a valid portfolio.
        '''
        new_array = array**2
        sum_squares = new_array.sum(axis = -1, keepdims = True)
        
        zero_rows = sum_squares[:, 0] == 0
        new_array[zero_rows] = 1
        sum_squares[zero_rows] = new_array.shape[-1]

        return new_array/sum_squares

    def alpha(self, array: np.ndarray) -> np.ndarray:
        '''
        Takes a portfolio array as an input, and outputs the alpha value of each portfolio in that matrix,
        based on the alpha values of each individual investment in that portfolio.
        '''
        return array.dot(self.alphas)

    def beta(self, array: np.ndarray) -> np.ndarray:
        '''
        Takes a portfolio array as an input, and outputs the beta value of each portfolio in that matrix,
        based on the beta values of each individual investment in that portfolio.
        '''
        return array.dot(self.betas)

    def allocations_by_category(self, array: np.ndarray) -> np.ndarray:
        '''
        Takes a portfolio array as an input, and outputs the allocations of each portfolio in each allocation
        category for that matrix, based on the category of each individual investment in that portfolio.
        '''
        return array.dot(self.category_matrix)

    def alpha_error(self, array: np.ndarray, apply_weight: bool = True) -> np.ndarray:
        '''
        Calculates and returns the amount of error for each portfolio in the given portfolio array based
        solely on its alpha.
        
        Error for alpha is calculated based on the exponential: E_alpha = e^(-alpha)
        '''
        if apply_weight:
            return self.alpha_error(array, False) * self.alpha_weight
        return np.exp(-self.alpha(array))

    def beta_error(self, array: np.ndarray, apply_weight: bool = True) -> np.ndarray:
        '''
        Calculates and returns the amount of error for each portfolio in the given portfolio array based
        solely on its beta, as compared to the target beta value.
        
        Error for beta is calculated based on the squared difference between portfolio beta and the target beta:
        E_beta = (beta - target_beta)^2
        '''
        if apply_weight:
            return self.beta_error(array, False) * self.beta_weight
        return (self.beta(array) - self.target_beta)**2

    def allocation_error(self, array: np.ndarray, apply_weight: bool = True) -> np.ndarray:
        '''
        Calculates and returns the amount of error for each portfolio in the given portfolio array based
        solely on its category allocations, as compared to the target category allocations.

        Error for allocations is calculated based on the sum of squared differences between the portfolio's
        allocations and the desired allocations, across all category types:
        E_alloc = sum over every category (allocation_in_category - target_allocation_in_category)^2
        '''
        if apply_weight:
            return self.allocation_error(array, False) * self.allocation_weight
        return ((self.allocations_by_category(array) - self.target_alloc_by_category)**2).sum(axis = -1)

    def total_error(self, array: np.ndarray) -> float:
        '''
        Given a portfolio array, calculates and returns the sum of all error types for that array, including weights.
        See alpha_error, beta_error, and allocation_error for more details on how each part operates individually.
        '''
        return self.alpha_error(array) + self.beta_error(array) + self.allocation_error(array)

    def partials(self, array: np.ndarray) -> np.ndarray:
        '''
        For a given portfolio array, calculates its gradient based on the error function provided by total_error.
        Note that this method assumes the provided array has *not* been normalized, so it first converts the array
        to a normalized array before computing the partial derivatives.

        See the readme for more details on how this is calculated.
        '''
        p, n = array.shape
        S = (array**2).sum(axis = -1)

        normalized = PortfolioModel.normalize(array)

        alpha = self.alpha(normalized)
        beta = self.beta(normalized)
        allocations = self.allocations_by_category(normalized)

        # These are the multiplicative constants for each portion of the error function: alpha beta, and allocation.
        alpha_delta = -e**(-alpha) * self.alpha_weight * 2 / S
        beta_delta = self.beta_weight * 4 * (beta - self.target_beta) / S
        alloc_delta = self.allocation_weight * 4 / S

        alloc_diffs = allocations - self.target_alloc_by_category

        self_beta_mat = np.broadcast_to(self.betas, (p,n))
        beta_mat = np.broadcast_to(beta.reshape(p, 1), (p,n))

        self_alpha_mat = np.broadcast_to(self.alphas, (p,n))
        alpha_mat = np.broadcast_to(alpha.reshape(p, 1), (p,n))

        # Now each portion of the parial derivatives are added together.
        partial = beta_delta.reshape(p, 1) * array * (self_beta_mat - beta_mat)
        partial += alpha_delta.reshape(p, 1) * array * (self_alpha_mat - alpha_mat)
        partial += alloc_delta.reshape(p, 1) * array * (alloc_diffs[:, self.category_indices] - (alloc_diffs * allocations).sum(axis = 1).reshape(p, 1))

        return partial

    def portfolio_string(self, array: np.ndarray, round_to: int = 2) -> str:
        '''
        Takes a single portfolio (one dimensional) array, and returns a string representation of its individual
        investment allocations.
        '''
        percents = np.round(array * 100, round_to)
        where_nonzero = percents != 0
        percents = percents[where_nonzero].astype(str) + "% "
        names = self.names[where_nonzero]
        return ', '.join(percents + names)

    def portfolio_allocation_string(self, array: np.ndarray, round_to: int = 2) -> str:
        '''
        Takes a single portfolio (one dimensional) array, and returns a string representation of its investment
        category allocations.
        '''
        percents = np.round(self.allocations_by_category(array) * 100, round_to)
        where_nonzero = percents != 0
        percents = percents[where_nonzero].astype(str) + "% "
        categories = self.category_list[where_nonzero]
        return ', '.join(percents + categories)

    def desired_allocations_str(self) -> str:
        '''
        Returns the target investment allocations, formatted nicely into a string.
        '''
        where_nonzero = self.target_alloc_by_category != 0
        percents = self.target_alloc_by_category[where_nonzero]*100
        percents = percents.astype(str) + "% "
        categories = self.category_list[where_nonzero]
        return ', '.join(percents + categories)
