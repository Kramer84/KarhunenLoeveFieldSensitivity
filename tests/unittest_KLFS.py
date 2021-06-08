import _aggregatedKarhunenLoeveResults as aklr
import _karhunenLoeveGeneralizedFunctionWrapper as klgfw
import _karhunenLoeveSobolIndicesExperiment as klsie
import _sobolIndicesFactory as sif

import openturns as ot
import numpy as np

import unittest


## Dummy Function taking as an input a 2D field, a 1D field and a scalar
def dummyFunction2Wrap(field_10x10, field_100x1, scalar_0):
    ## Function doing some operation on the 2 field and a scalar and returning a field
    outDim = 1
    NElem = [10]
    mesher = ot.IntervalMesher(NElem)
    lowerBound = [0]
    upperBound = [10]
    interval = ot.Interval(lowerBound,upperBound)
    mesh = mesher.build(interval)
    outField = ot.Field(mesh,[[0]]*mesh.getVerticesNumber())

    for i in range(10):
        for j in range(10):
            if field_10x10[i][0]*field_100x1[i+j][0]>scalar_0[0][0]:
                outField.setValueAtIndex(i, [field_10x10[i][0]*field_100x1[(i+1)*(j+1)-1][0] - scalar_0[0][0]])
            else :
                outField.setValueAtIndex(i, [(field_10x10[j][0] - scalar_0[0][0])/field_100x1[(i+1)*(j+1)-1][0]])

    return outField

##  Now we need 2 stochastic processes and a distribution.
## Let's first define the 2 meshes on which the processes are constructed
### The 2D Mesh :

intervals_2D = [10, 10]
low_bounds_2D = [0,0]
upper_bounds_2D = [10,10]
interval_mesher_2D = ot.IntervalMesher(intervals_2D)
grid_interval_2D = ot.Interval(low_bounds_2D, upper_bounds_2D)
mesh_2D = interval_mesher_2D.build(grid_interval_2D)

### The 1D Mesh :

elems_1D = [99]
mesher_1D = ot.IntervalMesher(elems_1D)
lowerBound_1D = [0]
upperBound_1D = [100]
interval_1D = ot.Interval(lowerBound_1D,upperBound_1D)
mesh_1D = mesher_1D.build(interval_1D)

print('The 1D mesh is\n', mesh_1D,'\n')
print('The 2D mesh is\n', mesh_2D,'\n')

## Now let's define the covariance models of both processes.
## we will for both use exponential models with the same base parameters
### The 1D cova model:
model_1D = ot.ExponentialModel([10], [1])

### The 2D cova model:
model_2D = ot.ExponentialModel([1,1], [1])

##Now finally let's get our two processes and the ditribution.
### The 1D Gaussian process
process_1D = ot.GaussianProcess(model_1D, mesh_1D)

### The 2D Gaussian process
process_2D = ot.GaussianProcess(model_2D, mesh_2D)

### The normal distribution:
scalar_distribution = ot.Normal()

## Now the we have our processes and distributions, let's first evaluate the function
## without any use of a wrapper or anything.
#### First get fields and samples from our processes and distributions
ot.RandomGenerator_SetSeed(888)
field_1D = process_1D.getRealization()
field_2D = process_2D.getRealization()
scalar_0 = [scalar_distribution.getRealization()]


print('For field 1D:\n',field_1D,'\n')
print('For field 2D:\n',field_2D,'\n')
print('For scalar :\n',scalar_0,'\n')
output_dummy_0 = dummyFunction2Wrap(field_2D, field_1D, scalar_0)

print('Output is:\n',output_dummy_0)

## Now that we have our processes defined, our realizations and the corresponding output
## we can create our aggregated object, wrap our function, and check if it behaves accordingly
### For that we will first have to do the Karhunen-Loeve decomposition of the processes.
algo_kl_process_1D = ot.KarhunenLoeveP1Algorithm(mesh_1D, process_1D.getCovarianceModel())
algo_kl_process_1D.run()
kl_results_1D = algo_kl_process_1D.getResult()

algo_kl_process_2D = ot.KarhunenLoeveP1Algorithm(mesh_2D, process_2D.getCovarianceModel())
algo_kl_process_2D.run()
kl_results_2D = algo_kl_process_2D.getResult()

### Now let's compose our Karhunen Loeve Results and our distributions.
composedKLResultsAndDistributions = aklr.AggregatedKarhunenLoeveResults([kl_results_2D, kl_results_1D, scalar_distribution])

### Now let's see if we manage to project and lift the realizations we had before.
realizationFields = [field_2D, field_1D, ot.Field(ot.Mesh(), [scalar_0[0]])]
projectedCoeffs = composedKLResultsAndDistributions.project(realizationFields)
print('Projected coefficients are :', projectedCoeffs)
liftedFieldsO  = composedKLResultsAndDistributions.liftAsField(projectedCoeffs)
print('Lifted fields are :', liftedFieldsO)

### Now let's use our function wrapper and see if we get the same results!
dummyWrapper = klgfw.KarhunenLoeveGeneralizedFunctionWrapper(composedKLResultsAndDistributions,
                                                        dummyFunction2Wrap, None, 1)

print('testing call:')
dummyWrapper(projectedCoeffs)


class TestComposeAndWrap(unittest.TestCase):
    def testLiftAndProject(self, field_1D = field_1D,
                           field_2D = field_2D,
                           scalar_0 = scalar_0,
                           liftedFieldsO = liftedFieldsO):
        print('Checking if the lifted fields are the same that the ones that were projected')
        self.assertTrue(np.allclose(np.array(field_2D), np.array(liftedFieldsO[0])))
        self.assertTrue(np.allclose(np.array(field_1D), np.array(liftedFieldsO[1])))
        self.assertTrue(np.allclose(np.array(scalar_0), np.array(liftedFieldsO[2])))
        print('LIfting is OK')
        print('THIS MEANS THAT THE FIRST OBJECT IS OK...')
        print('THIS MEANS THAT THE REASON FOR OUR ERROR IS UNKNOWN')
        print(":''(")
        print(":''(")
        print(":''(")
        print(":''(")
        print(":''(")

    def testEvaluateWithWrapper(self,coeffs = projectedCoeffs):
        global output_dummy_0
        output = dummyWrapper(coeffs)
        print(' output is:',output)
        print('output was:',output_dummy_0)
        self.assertTrue(np.allclose(np.array(output_dummy_0), np.array(output)))
        print('Function wrapper seems to be also OK... ')



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

# Definition of centered normal variable
N05 = ot.Normal(0,5)

# Definition of centered normal variable
N55 = ot.Normal(5,5)







def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)

class TestAggregatedKarhunenLoeve(unittest.TestCase):

    def setUp(self):
        self.AKLR0 = aklr.AggregatedKarhunenLoeveResults([results, N05])
        self.AKLR0.setLiftWithMean(False)

        self.AKLR1 = aklr.AggregatedKarhunenLoeveResults([results, N55])
        self.AKLR1.setMean(0,1000)
        self.AKLR1.setLiftWithMean(True)

        print('\nCreated aggregated karhunen loeve object')

    def testBaseFunctionalityNoMean(self):
        n_modes = self.AKLR0.getSizeModes()
        randVect = ot.ComposedDistribution([ot.Normal()]*n_modes)
        ot.RandomGenerator_SetSeed(6813484786)
        randPoint = randVect.getRealization()
        print('random point is', randPoint)
        ot.RandomGenerator_SetSeed(68213484786)
        randSample = randVect.getSample(10)
        #func_pt = self.AKLR0.lift(randPoint)
        field_pt = self.AKLR0.liftAsField(randPoint)
        smpl_pt = self.AKLR0.liftAsSample(randPoint)
        procsamp_samp = self.AKLR0.liftAsProcessSample(randSample)

        print(smpl_pt[0])
        coeffs_field_pt = self.AKLR0.project(field_pt)
        coeffs_smpl_pt = self.AKLR0.project(smpl_pt)
        coeffs_procsamp_samp = self.AKLR0.project(procsamp_samp)

        #self.assertEqual(randPoint, coeffs_func_pt)
        print('The modes are as follows,',self.AKLR0.__mode_count__)
        for i, (a,b) in enumerate(list(zip(list(randPoint), list(coeffs_field_pt)))):
            msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
            print('a_field:',a,'b_field:',b)
            self.assertAlmostEqual(a, b, 7, msg)
        print('From coeffs to fields to coeffs OK')

        for i, (a,b) in enumerate(list(zip(list(randPoint), list(coeffs_smpl_pt)))):
            msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
            print('a_sample:',a,'b_sample:',b)
            self.assertAlmostEqual(a, b, 7, msg)
        print('From coeffs to samples to coeffs OK')

        for j in range(randSample.getSize()):
            pt_j = randSample[j]
            pt_proc = coeffs_procsamp_samp[j]
            for i, (a,b) in enumerate(list(zip(list(pt_j), list(pt_proc)))):
                msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
                self.assertAlmostEqual(a, b, 7, msg)
        print('From coeffs to process samples to coeffs OK')

        print('Tests Passed without mean!')

    def testBaseFunctionalityMean(self):
        n_modes = self.AKLR1.getSizeModes()
        randVect = ot.ComposedDistribution([ot.Normal()]*n_modes)
        ot.RandomGenerator_SetSeed(68173484786)
        randPoint = randVect.getRealization()
        print('random point is', randPoint)
        ot.RandomGenerator_SetSeed(681348445786)
        randSample = randVect.getSample(10)
        #func_pt = self.AKLR1.lift(randPoint)
        field_pt = self.AKLR1.liftAsField(randPoint)
        smpl_pt = self.AKLR1.liftAsSample(randPoint)
        procsamp_samp = self.AKLR1.liftAsProcessSample(randSample)

        print(smpl_pt[0])
        coeffs_field_pt = self.AKLR1.project(field_pt)
        coeffs_smpl_pt = self.AKLR1.project(smpl_pt)
        coeffs_procsamp_samp = self.AKLR1.project(procsamp_samp)

        #self.assertEqual(randPoint, coeffs_func_pt)
        print('The modes are as follows,',self.AKLR1.__mode_count__)
        for i, (a,b) in enumerate(list(zip(list(randPoint), list(coeffs_field_pt)))):
            msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
            print('a_field:',a,'b_field:',b)
            self.assertAlmostEqual(a, b, 7, msg)
        print('From coeffs to fields to coeffs OK')

        for i, (a,b) in enumerate(list(zip(list(randPoint), list(coeffs_smpl_pt)))):
            msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
            print('a_sample:',a,'b_sample:',b)
            self.assertAlmostEqual(a, b, 7, msg)
        print('From coeffs to samples to coeffs OK')

        for j in range(randSample.getSize()):
            pt_j = randSample[j]
            pt_proc = coeffs_procsamp_samp[j]
            for i, (a,b) in enumerate(list(zip(list(pt_j), list(pt_proc)))):
                msg = 'assertAlmostEqual Failed for element {} of list, with values {} and {}'.format(i,a,b)
                self.assertAlmostEqual(a, b, 7, msg)
        print('From coeffs to process samples to coeffs OK')

        print('Tests Passed!')

#class DummyFuncResults :
#    dim = 25
#    size = 1000
#    np.random.seed(125)
#    ListOfPoints = [ot.Point(np.random.random(size))]*dim
#    BigSample =  ot.Sample(np.random.random((size,dim)))
#    NumpySample = np.random.random((size,dim))
#    ListNumpySamples = [np.random.random((size,dim)),
#                        np.random.random((size,dim-2)),
#                        np.random.random((size,dim+3)),
#                        np.random.random((size,dim-15))]




#lass Test_karhunenLoeveGeneralizedFunctionWrapper(unittest.TestCase):#

#    def setUp(self):
#        self.pts = DummyFuncResults.ListOfPoints
#        self.smp = DummyFuncResults.BigSample
#        self.npsmp = DummyFuncResults.NumpySample
#        self.nplst = DummyFuncResults.ListNumpySamples#
#

#    def testTransformations(self):
#        try :
#            X = klgfw.convertIntoProcessSample(self.pts)
#        except :#

#        print("For self.pts :",X )
#        try :
#            X = klgfw.convertIntoProcessSample(self.smp)
#        except :#

#        print("For self.smp :",X )
#        try :
#            X = klgfw.convertIntoProcessSample(self.npsmp)
#        except :#

#        print("For self.npsmp :", X)
#        try :
#            X = klgfw.convertIntoProcessSample(self.nplst)
#        except :#

#        print("For self.nplst :", X)








if __name__ == '__main__':
    unittest.main()
