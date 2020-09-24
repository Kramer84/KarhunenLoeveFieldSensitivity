import openturns as ot
import SobolIndicesAlgorithmField as siaf
import unittest 
from copy import deepcopy

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

def all_same(items):
    #Checks if all items of a list are the same
    return all(x == items[0] for x in items)

class TestAggregatedKarhunenLoeve(unittest.TestCase):

    def setUp(self):
        self.struct0 = KLResultLists.getKLResultHomogenous()
        self.struct1 = KLResultLists.getKLresultAggregated()
        self.struct2 = KLResultLists.getKLResultHeterogenous()
        self.AggrKL0 = siaf.AggregatedKarhunenLoeveResults(self.struct0[0])
        self.AggrKL0.setName('AggKL0')
        print('\n',self.AggrKL0._modesPerProcess)
        print('\n',self.AggrKL0._modeDescription)
        self.AggrKL1 = siaf.AggregatedKarhunenLoeveResults(self.struct1[0])
        self.AggrKL1.setName('AggKL1')
        print('\n',self.AggrKL1._modesPerProcess)
        print('\n',self.AggrKL1._modeDescription)
        self.AggrKL2 = siaf.AggregatedKarhunenLoeveResults(self.struct2[0])
        self.AggrKL2.setName('AggKL2')
        print('\n',self.AggrKL2._modesPerProcess)
        print('\n',self.AggrKL2._modeDescription)
        print('\n\n     ################### SETUP DONE ####################" \n\n\n')

    def test_Tegrity(self):
        print('First verifiying data intergrity')
        self.assertIsNotNone(self.struct0[1])
        self.assertIsNotNone(self.struct1[1])
        self.assertIsNotNone(self.struct2[1])

        self.assertIsNotNone(self.struct0[2])
        self.assertIsNotNone(self.struct1[2])
        self.assertIsNotNone(self.struct2[2])

        self.assertEqual(self.AggrKL0.getName(), 'AggKL0')
        self.assertEqual(self.AggrKL1.getName(), 'AggKL1')
        self.assertEqual(self.AggrKL2.getName(), 'AggKL2')
        print("Welcome to tegrity farms!")

    def test_projections(self):

        #realisations
        AggrKL2ProjRea = self.AggrKL2.project(self.struct2[1])

        #Samples
        AggrKL2ProjSamp = self.AggrKL2.project(self.struct2[2])

################################################################
        print('''
Testing Karhunen-Loeve result gotten from aggregated process.''')
        print('Projectig samples and fields:')
        AgKL1PrjRea_n = self.AggrKL1.project(self.struct1[1])
        AgKL1PrjSamp_n = self.AggrKL1.project(self.struct1[2])
        print('Projecting with old object:')
        AgKL1PrjRea_o = self.struct1[0].project(self.struct1[1])
        AgKL1PrjSamp_o = self.struct1[0].project(self.struct1[2])
        print('checking behaviour:')

        self.assertEqual(AgKL1PrjRea_n ,AgKL1PrjRea_o)
        self.assertEqual(AgKL1PrjSamp_n ,AgKL1PrjSamp_o)

        print('     OK ! ')
        print('\n\nNow checking lifting - aggregated process:')
        fldAggKL1 = self.AggrKL1.liftAsField(AgKL1PrjRea_n)
        fldAggKL1_o = self.struct1[0].liftAsField(AgKL1PrjRea_n)

        self.assertEqual( fldAggKL1[0], fldAggKL1_o)
        print(' lifting sample - aggregated process')
        proSmpAggKL1 = self.AggrKL1.liftAsField(AgKL1PrjSamp_n)
        #self.assertEqual(proSmpAggKL1, self.struct1[2])


################################################################
        print('''\n\n
Testing Karhunen-Loeve result gotten from list of homogen processes.\n''')
        print('Projectig samples and fields:')
        AgKL0PrjRea_n = self.AggrKL0.project(self.struct0[1])
        AgKL0PrjSamp_n = self.AggrKL0.project(self.struct0[2])

        print('Projecting with old object:')
        AgKL0PrjRea_o = [self.struct0[0][i].project(self.struct0[1][i]) for i in range(len(self.struct0[0]))]
        AgKL0PrjSamp_o =[self.struct0[0][i].project(self.struct0[2][i]) for i in range(len(self.struct0[0]))]
        print('checking behaviour:')
        print('Short correction:')
        AgKL0PrjRea_o = [item for sublist in AgKL0PrjRea_o for item in sublist]
        AgKL0PrjSamp_o = [item for sublist in AgKL0PrjSamp_o for item in sublist]

        print('AgKL0PrjSamp_n :\n',AgKL0PrjSamp_n )
        self.assertEqual(AgKL0PrjRea_n ,AgKL0PrjRea_o)
        #self.assertEqual(AgKL0PrjSamp_n ,AgKL0PrjSamp_o)

        print('  OK ! ')
        print('\n\nNow checking lifting - aggregated process:')
        fldAggKL0 = self.AggrKL0.liftAsField(AgKL0PrjRea_n)
        smpAggKL0 = self.AggrKL0.liftAsSample(AgKL0PrjSamp_n)
        proSmpAggKL0 = self.AggrKL0.liftAsField(AgKL0PrjSamp_n)

        
        print('smpAggKL0',smpAggKL0)
        print('self.struct0[2]',self.struct0[2])

        print('''\n\n
Testing Karhunen-Loeve result gotten from list of heterogen processes.\n''')
        print('Projectig samples and fields:')
        AgKL2PrjRea_n = self.AggrKL2.project(self.struct2[1])
        AgKL2PrjSamp_n = self.AggrKL2.project(self.struct2[2])
        print('AgKL2PrjSamp_n :\n',len(self.struct2[2]) )


        print('Projecting with old object:')
        AgKL2PrjRea_o = [self.struct2[0][i].project(self.struct2[1][i]) for i in range(len(self.struct2[0]))]
        AgKL2PrjSamp_o =[self.struct2[0][i].project(self.struct2[2][i]) for i in range(len(self.struct2[0]))]
        print('checking behaviour:')
        print('Short correction:')
        AgKL2PrjRea_o = [item for sublist in AgKL2PrjRea_o for item in sublist]
        AgKL2PrjSamp_o = [item for sublist in AgKL2PrjSamp_o for item in sublist]

        self.assertEqual(AgKL2PrjRea_n ,AgKL2PrjRea_o)
        #self.assertEqual(AgKL0PrjSamp_n ,AgKL0PrjSamp_o)

        print('  OK ! ')
        print('\n\nNow checking lifting - aggregated process:')
        fldAggKL2 = self.AggrKL2.liftAsField(AgKL2PrjRea_n)
        smpAggKL2 = self.AggrKL2.liftAsSample(AgKL2PrjSamp_n)
        proSmpAggKL2 = self.AggrKL2.liftAsField(AgKL2PrjSamp_n)

        
        print('smpAggKL0',smpAggKL2)
        print('self.struct0[2]',self.struct2[2])



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