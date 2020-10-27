import openturns as ot
import _aggregatedKarhunenLoeveResults as siaf
import unittest
from copy import deepcopy




# Defining a 1D process.
# first define mesh
dimension = 1
NElem = [100]
mesher = ot.IntervalMesher(NElem)
lowerBound = [0]
upperBound = [1000]
interval = ot.Interval(lowerBound,upperBound)
mesh = mesher.build(interval)

# then define covariance model
amplitude0 = [100]*dimension
scale0 = [300]*dimension
nu0 = 4.5
model0 = ot.MaternModel(scale0, amplitude0, nu0)

# then define the stochastic process
process = ot.GaussianProcess(model0, mesh)

# get some realizations and a sample
ot.RandomGenerator_SetSeed(11111)
field1D = process.getRealization() #FIELD BASE

ot.RandomGenerator_SetSeed(11111)
sample1D = process.getSample(10) #SAMPLE BASE

# get the Karhunen Loeve decomposition of the mesh
algorithm = ot.KarhunenLoeveP1Algorithm(mesh, model0, 1e-3)
algorithm.run()
results = algorithm.getResult()   #### This is the object we will need !

#now let's project the field and the samples on the eigenmode basis
lifter = ot.KarhunenLoeveLifting(results)
projecter = ot.KarhunenLoeveProjection(results)

coeffField1D = projecter(field1D)
coeffSample1D = projecter(sample1D) #dimension of the coefficents, done internaly by our class but needed for comparison

fieldVals = lifter(coeffField1D)
sample_lifted = lifter(coeffSample1D)
field_lifted = ot.Field(lifter.getOutputMesh(),fieldVals)

# NOw we also define a random variable
N = ot.Normal(0,5)




def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)

class TestAggregatedKarhunenLoeve(unittest.TestCase):

    def setUp(self):
        self.AKLR = siaf.AggregatedKarhunenLoeveResults([results, N])
        self.AKLR.setLiftWithMean(True)
        ## Here we create our new object
        print('\nCreated aggregated karhunen loeve object')

    def testBaseFunctionality(self):
        n_modes = self.AKLR.getSizeModes()
        randVect = ot.ComposedDistribution([ot.Normal()]*n_modes)
        ot.RandomGenerator_SetSeed(6813484786)
        randPoint = randVect.getRealization()
        print('random point is', randPoint)
        ot.RandomGenerator_SetSeed(6813484786)
        randSample = randVect.getSample(10)
        #func_pt = self.AKLR.lift(randPoint)
        field_pt = self.AKLR.liftAsField(randPoint)
        print('field_pt[1]=',field_pt[1])
        smpl_pt = self.AKLR.liftAsSample(randPoint)
        procsamp_samp = self.AKLR.liftAsProcessSample(randSample)

        print(smpl_pt[0])
        coeffs_field_pt = self.AKLR.project(field_pt)
        coeffs_smpl_pt = self.AKLR.project(smpl_pt)
        coeffs_procsamp_samp = self.AKLR.project(procsamp_samp)

        #self.assertEqual(randPoint, coeffs_func_pt)
        print('The modes are as follows,',self.AKLR.__mode_count__)
        for i, (a,b) in enumerate(list(zip(list(randPoint), list(coeffs_field_pt)))):
            msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
            print('a:',a,'b:',b)
            self.assertAlmostEqual(a, b, 7, msg)
        self.assertAlmostEqual(list(randPoint), list(coeffs_smpl_pt))
        self.assertAlmostEqual(randSample, coeffs_procsamp_samp)

        print('is OK!')


if __name__ == '__main__':
    unittest.main()

"""

import SobolIndicesAlgorithmField as siaf
import SobolIndicesAlgorithmField_unittest as siaft

KLRes, Fld, Smp = siaft.KLResultLists.getKLresultAggregated()
test = siaf.AggregatedKarhunenLoeveResults(KLRes)

test.project(Fld)
test.project(Smp)

coefFld = test.project(Fld)
coefSmp = test.project(Smp)

Fld0 = test.liftAsField(coefFld)
Smp0 = test.liftAsSample(coefSmp)

"""
"""


class testVals1D :
    dim       = 1
    amplitude = [1]
    scale     = [3]*dim
    nu        = 5
    model0    = ot.MaternModel(scale, amplitude, nu)
    model1    = ot.AbsoluteExponential(scale, amplitude)
    model2    = ot.ExponentialModel(scale, amplitude)
    t0        = 0
    step      = 1
    N         = 11
    grid      = ot.RegularGrid(t0, step, N)
    func      = ot.SymbolicFunction(['x'],['1'])
    trendFunc = ot.TrendTransform(func, grid)
    process0  = ot.GaussianProcess(deepcopy(trendFunc), deepcopy(model0), deepcopy(grid))
    process0.setName('P1D0')
    process1  = ot.GaussianProcess(deepcopy(trendFunc), deepcopy(model1), deepcopy(grid))
    process1.setName('P1D1')
    process2  = ot.GaussianProcess(deepcopy(trendFunc), deepcopy(model2), deepcopy(grid))
    process2.setName('P1D2')

class testVals2D :
    dim       = 2
    amplitude = [1]
    scale     = [3]*dim
    nu        = 5
    model0    = ot.MaternModel(scale, amplitude, nu)
    model1    = ot.AbsoluteExponential(scale, amplitude)
    model2    = ot.ExponentialModel(scale, amplitude)
    shape     = [11, 11]
    mesher    = ot.IntervalMesher(shape)
    lowBound  = [0., 0.]
    upperBnd  = [10. ,10.]
    interval  = ot.Interval(lowBound, upperBnd)
    meshBox   = mesher.build(interval)
    func      = ot.SymbolicFunction(['x','Y'],['1'])
    trendFunc = ot.TrendTransform(func, meshBox)
    process0  = ot.GaussianProcess(deepcopy(trendFunc), deepcopy(model0), deepcopy(meshBox))
    process0.setName('P2D0')
    process1  = ot.GaussianProcess(deepcopy(trendFunc), deepcopy(model1), deepcopy(meshBox))
    process1.setName('P2D1')
    process2  = ot.GaussianProcess(deepcopy(trendFunc), deepcopy(model2), deepcopy(meshBox))
    process2.setName('P2D2')

class KLResultLists:
    @staticmethod
    def getKLResultHomogenous():
        grid  = deepcopy(testVals1D.grid)
        P0    = deepcopy(testVals1D.process0)
        P1    = deepcopy(testVals1D.process1)
        P2    = deepcopy(testVals1D.process2)
        algo0 = ot.KarhunenLoeveP1Algorithm(grid, P0.getCovarianceModel(), 1e-3)
        algo1 = ot.KarhunenLoeveP1Algorithm(grid, P1.getCovarianceModel(), 1e-3)
        algo2 = ot.KarhunenLoeveP1Algorithm(grid, P2.getCovarianceModel(), 1e-3)
        algo0.run()
        algo1.run()
        algo2.run()
        R0 = algo0.getResult()
        R0.setName('R0')
        R1 = algo1.getResult()
        R1.setName('R1')
        R2 = algo2.getResult()
        R2.setName('R2')
        ot.RandomGenerator.SetSeed(128)
        [REA0, REA1, REA2] = [P0.getRealization(), P1.getRealization(), P2.getRealization()]
        [SMP0, SMP1, SMP2] = [P0.getSample(5), P1.getSample(5), P2.getSample(5)]
        return [R0, R1, R2], [REA0, REA1, REA2], [SMP0, SMP1, SMP2]

    @staticmethod
    def getKLresultAggregated():
        grid = deepcopy(testVals1D.grid)
        P0   = deepcopy(testVals1D.process0)
        P1   = deepcopy(testVals1D.process1)
        P2   = deepcopy(testVals1D.process2)
        aggregated = ot.AggregatedProcess([P0, P1, P2])
        algo = ot.KarhunenLoeveP1Algorithm(grid, aggregated.getCovarianceModel(), 1e-3)
        algo.run()
        R = algo.getResult()
        R.setName('R')
        ot.RandomGenerator.SetSeed(128)
        REA = aggregated.getRealization()
        SMP = aggregated.getSample(5)
        return R, REA, SMP

    @staticmethod
    def getKLResultHeterogenous():
        grid1d = deepcopy(testVals1D.grid)
        grid2d = deepcopy(testVals2D.meshBox)
        P0   = deepcopy(testVals1D.process0)
        P1   = deepcopy(testVals2D.process1)
        P2   = deepcopy(testVals1D.process2)
        algo0 = ot.KarhunenLoeveP1Algorithm(grid1d, P0.getCovarianceModel(), 1e-3)
        algo1 = ot.KarhunenLoeveP1Algorithm(grid2d, P1.getCovarianceModel(), 1e-3)
        algo2 = ot.KarhunenLoeveP1Algorithm(grid1d, P2.getCovarianceModel(), 1e-3)
        algo0.run()
        algo1.run()
        algo2.run()
        R0 = algo0.getResult()
        R0.setName('R0')
        R1 = algo1.getResult()
        R1.setName('R1')
        R2 = algo2.getResult()
        R2.setName('R2')
        ot.RandomGenerator.SetSeed(128)
        [REA0, REA1, REA2] = [P0.getRealization(), P1.getRealization(), P2.getRealization()]
        [SMP0, SMP1, SMP2] = [P0.getSample(5), P1.getSample(5), P2.getSample(5)]
        return [R0, R1, R2], [REA0, REA1, REA2], [SMP0, SMP1, SMP2]

"""
