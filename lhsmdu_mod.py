# To create an orthogonal Latin hypercube with uniform sampling of parameters.
# Author: Sahil Moza
# Date: Jan 21, 2016

''' This is an implementation of Latin Hypercube Sampling with Multi-Dimensional Uniformity (LHS-MDU) from Deutsch and Deutsch, "Latin hypercube sampling with multidimensional uniformity", Journal of Statistical Planning and Inference 142 (2012) , 763-772 

***Currently only for independent variables***
'''

#Date: Apr 29, 2020
#Code acceleration 
#Author:Kristof S.
#modified functions : #ModKS

from   __future__ import absolute_import, division, print_function, unicode_literals
import numpy 
import scipy
import   numba 

##### Default variables #####
scalingFactor   = 5 ## number > 1 (M) Chosen as 5 as suggested by the paper (above this no improvement.
numToAverage    = 2 ## Number of nearest neighbours to average, as more does not seem to add more information (from paper).
randSeed        = 42 ## Seed for the random number generator 
numpy.random.seed(randSeed) ## Seeding the random number generator.

def setRandomSeed(newRandSeed):
    global randSeed
    randSeed = newRandSeed
    numpy.random.seed(randSeed) ## Seeding the random number generator.

def createRandomStandardUniformMatrix(nrow, ncol): 
    ''' Creates a matrix with elements drawn from a uniform distribution in [0,1]'''
    return numpy.random.uniform(size=(nrow,ncol))     #ModKS #usage of numpy (25* faster)

@numba.jit(nopython=True, parallel = True)
def findUpperTriangularColumnDistanceVector(inputMatrix):
    ''' Finds the 1-D upper triangular euclidean distance vector for the columns of a matrix.'''
    #ModKS #assert ncol == inputMatrix.shape[1] => just say that ncol = inputMatrix.shape[1] and remove arg from function
    # accelerated with numba (really fast!)
    ncol = inputMatrix.shape[1]
    distance_1D = []
    for i in range(ncol-1):
        for j in range(i+1,ncol):
            realization_i, realization_j  = inputMatrix[:,i], inputMatrix[:,j]
            distance_1D.append(numpy.linalg.norm(realization_i - realization_j))
    return distance_1D

#@jit(nopython=True)
def createSymmetricDistanceMatrix(distance, nrow):
    ''' Creates a symmetric distance matrix from an upper triangular 1D distance vector.'''
    # Fast enough
    distMatrix                           = numpy.zeros((nrow,nrow))
    indices                              = numpy.triu_indices(nrow,k=1)
    distMatrix[indices]                  = distance
    distMatrix[(indices[1], indices[0])] = distance # Making symmetric matrix
    return distMatrix


def eliminateRealizationsToStrata(distance_1D, matrixOfRealizations, numSamples, numToAverage = numToAverage):
    ''' Eliminating realizations using average distance measure to give Strata '''

    numDimensions   = matrixOfRealizations.shape[0]
    numRealizations = matrixOfRealizations.shape[1]
    ## Creating a symmetric IxI distance matrix from the triangular matrix 1D vector.
    distMatrix      = createSymmetricDistanceMatrix(distance_1D, numRealizations)
 
    ## Finding columns from the realization matrix by elimination of nearest neighbours L strata are left.
    ## Using matrices is faster, and works with numba
    averageDistanceIdx  = list(numpy.arange(numRealizations).astype(int))
    averageDistanceData = list(numpy.zeros(numRealizations))
    ## using external function for numba
    averageDistanceIdx = fasterSorting(averageDistanceIdx, averageDistanceData, distMatrix, numSamples, numToAverage)
    # Creating the strata matrix to draw samples from.
    StrataMatrix = matrixOfRealizations[:,averageDistanceIdx]
    
    assert numSamples    == StrataMatrix.shape[1]
    assert numDimensions == StrataMatrix.shape[0]
    #print ( StrataMatrix )
    return StrataMatrix

def fasterSorting(averageDistanceIdx, averageDistanceData, distMatrix, numSamples, numToAverage):
    var = len(averageDistanceIdx)
    while(var>numSamples):
        for rowNum in averageDistanceIdx:
            meanAvgDist = numpy.divide(numpy.sum(numpy.sort(distMatrix[rowNum, averageDistanceIdx])[:numToAverage+1]), numToAverage)
            averageDistanceData[averageDistanceIdx.index(rowNum)] = meanAvgDist  # +1 to remove the zero index, appending averageDistance to list
        indexToDelete = averageDistanceData.index(min(averageDistanceData))
        del(averageDistanceIdx[indexToDelete]) 
        del(averageDistanceData[indexToDelete])
        var = len(averageDistanceIdx)
    return averageDistanceIdx

#@jit(nopython=True)
def inverseTransformSample(distribution, uniformSamples):
    ''' This function lets you convert from a standard uniform sample [0,1] to
    a sample from an arbitrary distribution. This is done by taking the cdf [0,1] of 
    the arbitrary distribution, and calculating its inverse to picking the sample."
    '''
    assert (isinstance(distribution, scipy.stats.rv_continuous) or isinstance(distribution, scipy.stats.rv_discrete) or isinstance(distribution,scipy.stats.distributions.rv_frozen))
    newSamples = distribution.ppf(uniformSamples)
    return newSamples
    
@numba.jit(nogil=True) 
def resample(matrixOfStrata):

    ''' Resampling function from the same strata'''
    numDimensions   = matrixOfStrata.shape[0]
    numSamples      = matrixOfStrata.shape[1]
    matrixOfSamples = numpy.empty((numDimensions, numSamples))
    # Creating Matrix of Samples from the strata ordering.
    for row in range(numDimensions):
        sortedIndicesOfStrata = numpy.argsort(numpy.ravel(matrixOfStrata[row,:]))
        # Generating stratified samples
        newSamples =  [ (float(x)/numSamples) + (numpy.random.random()/numSamples) for x in sortedIndicesOfStrata ]
        matrixOfSamples[:,row]=newSamples

    #assert numpy.min(matrixOfSamples)>=0.
    #assert numpy.max(matrixOfSamples)<=1.
    return matrixOfSamples

#@jit(nopython=True)
def sample(numDimensions, numSamples, scalingFactor=scalingFactor, numToAverage = numToAverage, randomSeed=randSeed):
    ''' Main LHS-MDU sampling function '''

    if not randomSeed == randSeed:
        setRandomSeed(randomSeed)

    ### Number of realizations (I) = Number of samples(L) x scale for oversampling (M)
    numRealizations      = scalingFactor*numSamples ## Number of realizations (I)
    ### Creating NxI realization matrix
    matrixOfRealizations =  createRandomStandardUniformMatrix(numDimensions, numRealizations)  #ModKS
    ### Finding distances between column vectors of the matrix to create a distance matrix.
    distance_1D          = findUpperTriangularColumnDistanceVector(matrixOfRealizations)       #ModKS
    
    ## Eliminating columns from the realization matrix, using the distance measure  to get a strata
    ## matrix with number of columns as number of samples requried.

    matrixOfStrata = eliminateRealizationsToStrata(distance_1D, matrixOfRealizations, numSamples)

    matrixOfSamples = resample(matrixOfStrata) 
    
    return matrixOfSamples
