import pandas as pd
import numpy as np
from scipy.special import expit
from scipy import optimize
from myclient.adjustment import *
import statsmodels.api as sm


class ShadowRecovery:
    """
    Class for recovering the causal effect under self-censoring outcome.
    """

    def __init__(
        self, A, Y, R_Y, Z, dataset, ignoreMissingness=False, useWrongBackdoorSet=False
    ):
        """
        Constructor for the class.

        Due to the completeness condition of using shadow variables, the outcome
        cannot have more possible values than the treatment. For this implementation,
        A is a binary variable; hence, Y should be a binary variable as well.

        A, Y, and R_Y should be strings representing the names of the
        treatment, outcome, and missingness indicator for the outcome, respectively,
        as represented in the pandas dataframe dataset.

        Z should be a list of strings representing the names of the variables in the
        adjustment set as represented in dataset.

        dataset should be a dataset already with missing values for the outcome where
        missing values of the outcome are represented by -1.
        """
        self.A = A
        self.Y = Y
        self.R_Y = R_Y
        self.Z = Z
        self.dataset = dataset
        self.size = len(self.dataset)

        # drop all rows of data where R_Y=0
        self.subset_data = self.dataset[self.dataset[R_Y] == 1]

        # initialize list for the parameters of outcome
        self.paramsY = self._initializeParametersGuess()

        self.ignoreMissingness = ignoreMissingness
        self.useWrongBackdoorSet = useWrongBackdoorSet

    def _initializeParametersGuess(self):
        """
        Initialize a guess for the parameters in shadow recovery. The guesses are 0.0 for all the
        alpha parameters and 1.0 for the gamma parameter.
        """
        guess = []

        # when Z has k variables, there should be k+1 parameters
        # draw a number from the normal distribution between 0 and 1 as the initial guess
        for i in range(len(self.Z)):
            guess.append(np.random.uniform(0, 1, 1))
        guess.append(np.random.uniform(0, 1, 1))

        return guess

    def _estimatingEquations(self, params):
        """
        Define the estimating equations for shadow recovery.
        There are two cases: when Z is empty and when Z is non-empty.
        """
        # outputs represent the outputs of this functional
        outputs = []

        # p(R_Y = 1 | Y = 0) = the sum of all the parameters multiplied by each variable in Z
        # then put into an expit
        # when there are no parameters, pRX_X_0 = expit(0)
        pRY_Y_0 = expit(np.sum(params[0 : len(self.Z)] * self.dataset[self.Z], axis=1))
        # pRY_Y_0 = expit( np.sum(params[i]*self.dataset[self.Z[i]] for i in range(len(self.Z))) )
        # the final parameter is used to estmate OR(R_Y=0, Y)
        pRY = pRY_Y_0 / (
            pRY_Y_0
            + np.exp(params[len(self.Z)] * (1 - self.dataset[self.Y])) * (1 - pRY_Y_0)
        )

        # first k equations are each variable in Z individually
        for i in range(len(self.Z)):
            outputs.append(
                np.average(
                    (self.dataset[self.Z[i]] * self.dataset[self.R_Y]) / pRY
                    - (self.dataset[self.Z[i]])
                )
            )

        # final equation is the average of the shadow variable, in this case the treatment
        outputs.append(
            np.average(
                (np.average(self.dataset[self.A]) * self.dataset[self.R_Y]) / pRY
                - (np.average(self.dataset[self.A]))
            )
        )

        return outputs

    def _findRoots(self):
        """
        Estimate the roots for the treatment and outcome.
        """
        self.paramsY = optimize.root(
            self._estimatingEquations, self.paramsY, method="hybr"
        )

        for _ in range(5):
            if not self.paramsY.success:
                self.paramsY = optimize.root(
                    self._estimatingEquations, self.paramsY.x, method="hybr"
                )

        #print(self.paramsY.success)

        self.paramsY = self.paramsY.x
        #print(self.paramsY)

    def _propensityScoresRY(self, data):
        """
        Predict the propensity scores for the missingness of the outcome using the recovered
        parameters.
        """
        # p(R_Y = 1 | Y = 0)
        pRY_Y_0 = expit(
            sum(self.paramsY[i] * data[self.Z[i]] for i in range(len(self.Z)))
        )
        propensityScoresRY = pRY_Y_0 / (
            pRY_Y_0
            + np.exp(self.paramsY[len(self.Z)] * (1 - data[self.Y])) * (1 - pRY_Y_0)
        )

        return propensityScoresRY
