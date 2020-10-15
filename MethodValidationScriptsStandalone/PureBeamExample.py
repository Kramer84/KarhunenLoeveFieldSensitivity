import pickle
from joblib import Parallel, delayed

import numpy
import openturns as ot
from anastruct.fem.system import SystemElements, Vertex
from copy import deepcopy

counter = 0 

class _BeamBase(object):

    def __init__(self, mesh):
        global counter 
        self.mesh = mesh
        self.lElem = float((mesh.getUpperBound()[0]-mesh.getLowerBound()[0])/mesh.getSimplicesNumber())
        self.vertices = self.makeVerticesList()


    def makeVerticesList(self):
        n_elems = self.mesh.getVerticesNumber()
        vertices = numpy.arange(n_elems+1)*((self.mesh.getUpperBound() - self.mesh.getLowerBound()).norm()/(n_elems))
        vertex_list = [Vertex(vertices[i],0) for i in range(n_elems+1)]
        return vertex_list

    def batchEval(self, random_young_modulus, 
                  random_diameter, 
                  random_density, 
                  random_forcePos, 
                  random_forceNorm):
        var1, var2, var3, var4, var5, var6, var7 = random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm, self.vertices, self.lElem
        result_list = Parallel(n_jobs=-1, verbose=10)(
                        delayed(_BeamBase.experience)(
                            var1[i], var2[i], var3[i], var4[i], var5[i], var6, var7) for i in range(len(var5)))
        monteCarloResults_elem = numpy.stack(numpy.asarray(result_list)[...,0])
        monteCarloResults_node = numpy.stack(numpy.asarray(result_list)[...,1])
        monteCarloResults_glob = numpy.stack(numpy.asarray(result_list)[...,2])
        deflection = monteCarloResults_node[:,1,:]
        print('shape deflection: ', deflection.shape, ' should be [N,10X] something')
        vonMisesStress = self.getVonMisesStress(monteCarloResults_elem)
        maxDeflection = numpy.amax(numpy.abs(deflection), 1)
        print('deflection std deviation ',numpy.std(maxDeflection))
        return vonMisesStress, maxDeflection

    @staticmethod
    def singleEval(random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm, vertsList, lElem):
        result = _BeamBase.experience(numpy.squeeze(numpy.array(random_young_modulus)), 
                                     numpy.squeeze(numpy.array(random_diameter)),
                                     float(random_density),
                                     float(random_forcePos),
                                     float(random_forceNorm),
                                     vertsList,
                                     lElem)
        element_results, node_results, global_beamParams = result[0], result[1], result[2]
        diameter = element_results[2,...]
        moment   = element_results[4,...]
        shear    = element_results[3,...]
        inertia, area  = _BeamBase.moment_inertia_PlainRoundBeam(diameter)
        maxbendingStress  = _BeamBase.getMaximumBendingStress(moment, inertia, diameter)
        shearStress   = (4/3)*numpy.divide(shear, area)  
        vonMisesCriteria = numpy.sqrt(numpy.square(maxbendingStress) + numpy.multiply(numpy.square(shearStress), 3))
        maxDeflection = numpy.max(numpy.abs(node_results[1,...]))
        return  vonMisesCriteria, maxDeflection

    def getVonMisesStress(self,  monteCarloResults_elem=None):
        assert monteCarloResults_elem is not None
        diameter_MC          = monteCarloResults_elem[:,2,:]
        moment_MC            = monteCarloResults_elem[:,4,:]
        shear_MC             = monteCarloResults_elem[:,3,:]
        inertia_MC, area_MC  = _BeamBase.moment_inertia_PlainRoundBeam(diameter_MC)
        maxbendingStress_MC  = _BeamBase.getMaximumBendingStress(moment_MC, inertia_MC, diameter_MC)
        shearStress_MC       = (4/3)*numpy.divide(shear_MC, area_MC)  
        vonMisesCriteria     = numpy.sqrt(numpy.square(maxbendingStress_MC) + numpy.multiply(numpy.square(shearStress_MC), 3))
        return vonMisesCriteria

    @staticmethod
    def moment_inertia_PlainRoundBeam(D):
        # Function calculating the moment of inertia as well as the area of a beam given a diameter
        return numpy.divide(numpy.pi*numpy.power(D,4), 64), numpy.pi*numpy.power(numpy.divide(D,2),2)

    @staticmethod
    def getMaximumBendingStress(bendingMoment, inertia, diameter):
        assert bendingMoment.shape == inertia.shape , "same shape required"
        return numpy.multiply(numpy.divide(bendingMoment, inertia),-1*diameter/2)

### For multiprocessing purposes
    
    # On enleve le self de toutes les fonctions utilisées en multiprocessing, et en pase en méthode statique
    # car comme on posse en argument de la classe un objet NdGaussianProcessConstructor, le "self" ne peut plus
    # etre pickle, du coup en multiprocessing ca nous envoie une erreur, comme quoi un objet ne peut pas etre déserialisé
    # ce qui dans notre cas devrait être le "self" contenant une autre classe

    @staticmethod
    def experience( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element):
        '''Function that is used for multiprocessing the beam experience
        There are a lot of tests to make sure the codes don't crash
        '''
        global counter 
        youngModu     = numpy.squeeze(youngModu).astype(float)
        diam          = numpy.squeeze(diam).astype(float)
        n0_elems      = len(diam)
        density       = float(density)
        forcePosition = float(forcePosition)
        forceNorm     = float(forceNorm)
        BeamObj_MP, youngModu, diam, density, forcePosition, forceNorm = _BeamBase.instantiateRandomBeam_MP(youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element)
        counter += 1
        try :
            solution                  = BeamObj_MP.solve(force_linear = False,
                                                         verbosity = 50,
                                                         max_iter = 300, 
                                                         geometrical_non_linear = False,
                                                         naked = False)
            points_range              = numpy.asarray(BeamObj_MP.nodes_range('x'))
            elem_length_range         = points_range[1:]-(points_range[1]-points_range[0])/2
            deflection, shear, moment = _BeamBase.postprecess_beam_MP(BeamObj_MP)
            element_results           = numpy.vstack([elem_length_range,youngModu, diam, numpy.array(shear), numpy.array(moment)])
            node_results              = numpy.vstack([points_range, deflection])
            global_beamParams         = numpy.array([forceNorm, density, forcePosition])
            if counter % 47 == 0 :
                print('max deflection:',max(node_results[1,...]))
            return element_results, node_results, global_beamParams

        except Exception as e:
            print('there was an error in the stiffness matrix.\n','Error =',str(e),'\n filling gaps with numpy.nan values\n\n')
            #self.modelsWithErrors.append(BeamObj_MP)
            points_range        = numpy.array(BeamObj_MP.nodes_range('x'))
            elem_length_range   = points_range[1:]-(points_range[1]-points_range[0])/2
            shear = moment      = numpy.array([numpy.nan]*len(elem_length_range))
            deflection          = numpy.array([numpy.nan]*len(points_range))
            lens                = numpy.array([len(youngModu), len(diam), len(shear), len(moment)])
            assertionVect       = (lens == len(elem_length_range))
            if False not in assertionVect :
                print('Error but somehow corrected\n')
                element_results     = numpy.vstack([elem_length_range,youngModu, diam, shear, moment])
                node_results        = numpy.vstack([points_range, deflection])
                global_beamParams   = numpy.array([forceNorm, density, forcePosition])
                return element_results, node_results, global_beamParams

            if False in assertionVect :
                #here we replace all with nans, cause we gonna erease it anyways
                print('There was nothing more to do... Release the nan s!!! \n')
                elem_length_range   = youngModu  = diam =  shear =  moment = numpy.array([numpy.nan]*(n0_elems+1))
                points_range        = deflection = numpy.array([numpy.nan]*(n0_elems+2))
                element_results     = numpy.vstack([elem_length_range,youngModu, diam, shear, moment])
                node_results        = numpy.vstack([points_range, deflection])
                if type(forceNorm) == type(density)  == type(forcePosition) == float :
                    global_beamParams   = numpy.array([forceNorm, density, forcePosition])
                else :
                    global_beamParams   = numpy.array([numpy.nan]*3)
                return element_results, node_results, global_beamParams

    @staticmethod
    def instantiateRandomBeam_MP( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element):
        a, b, c=_BeamBase.anastruct_beam_MP(youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element)
        return a, b, c, density, forcePosition, forceNorm

    @staticmethod
    def anastruct_beam_MP( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element):
        system = SystemElements(EA = None, EI = None) # to make sure we delete the default values
        for k in range(len(vertex_list)-1):      # always a vertex more than element
            system.add_element(location = [vertex_list[k], vertex_list[k+1]], 
                                d = float(diam[k]),
                                E = float(youngModu[k]),
                                gamma = density)
        
        nodeID, system, random_diameter, random_young_modulus = _BeamBase.nodeChecksAndVertsInsertion_MP(system, forcePosition, diam, youngModu, vertex_list, l_element)
        system.point_load(nodeID, Fy = -forceNorm)   # in kN
        system.add_support_hinged(node_id=min(system.node_map.keys()))
        system.add_support_roll(node_id=system.id_last_node , direction='x')
        return system, random_young_modulus, random_diameter

    
    @staticmethod
    def nodeChecksAndVertsInsertion_MP( system, forcePosition, random_diameter, random_young_modulus, vertex_list, l_element):
        # Function to apply force at force position
        nearestNodeFposID = system.nearest_node('x', forcePosition)
        if abs(vertex_list[nearestNodeFposID].x - forcePosition)<1.:
            nodeID_F = nearestNodeFposID
            ## We add an other node next to it so we normalize the number of elements
            system.insert_node(element_id = nodeID_F+3, factor=0.5)
            random_diameter      = numpy.insert(random_diameter, nodeID_F+1, random_diameter[nodeID_F+1])
            random_young_modulus = numpy.insert(random_young_modulus, nodeID_F+1, random_young_modulus[nodeID_F+1])
            return nodeID_F, system, random_diameter, random_young_modulus
        else :
            elemNumber, factor = _BeamBase.getElemIdAndFactor_MP(forcePosition, l_element)
            system.insert_node(element_id = elemNumber, factor=factor)
            newNodeID = system.nearest_node('x',forcePosition)
            # as we inserted a node we have one more element so we add it 
            random_diameter      = numpy.insert(random_diameter, newNodeID-2, random_diameter[newNodeID-2])
            random_young_modulus = numpy.insert(random_young_modulus, newNodeID-2, random_young_modulus[newNodeID-2])
            return newNodeID, system, random_diameter, random_young_modulus

    @staticmethod
    def getElemIdAndFactor_MP( forcePosition, l_element):
        #print('forcePosition= ',forcePosition,'elemNumber= ',elemNumber, 'factor= ',factor)
        return int(forcePosition/l_element)+1, forcePosition/l_element - int(forcePosition/l_element) # +1 because anastruct counts elements from 1 

    @staticmethod
    def postprecess_beam_MP( BeamObj = None):
        # Function to get the different values of the simulation and do some plotting. This is not for the monte carlo simulation
        return [numpy.array(BeamObj.get_node_result_range('uy'), copy=False), 
                numpy.array(BeamObj.get_element_result_range('shear'), copy=False), 
                numpy.array(BeamObj.get_element_result_range('moment'), copy=False)]




from functools import partial

class PureBeam(object):

    def __init__(self, mesh):
        self.Beam = _BeamBase(mesh)
        self.vertsList = self.Beam.vertices
        self.lElem = self.Beam.lElem
        self.Means = [210000, 10, 7850, 500, 100]

    def singleEval(self, argList):
        inputs = list(self.convertSinglInputs(argList))
        inputs.extend([self.vertsList, self.lElem])
        return _BeamBase.singleEval(*inputs)

    def batchEval(self, argList):
        inputs = list(self.convertSinglInputs(argList))
        return self.Beam.batchEval(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]) 

    def partialBatch(self, partialArgs):
        pass


    ## Here we are going to define the function that is going to transform the list of processes 
    ## into a list of processes and random variables: 
    def addK2ProcSamp(self, PS, K):
        for i in range(PS.getSize()):
            PS[i]+=K 
        return deepcopy(PS)

    def addConst2List(self, L,K):
        for i in range(len(L)):
            L[i]+=K
        return deepcopy(L)
            
    def convertSinglInputs(self, inputList):
        '''To confert the fields into scalars
        '''
        if isinstance(inputList[0],ot.ProcessSample):
            field_E = inputList[0]
            field_D = inputList[1]
            var_Rho  = [inputList[2][i].getMax()[0] for i in range(inputList[2].getSize())]
            var_Fpos = [inputList[3][i].getMax()[0] for i in range(inputList[3].getSize())] 
            var_Fnor = [inputList[4][i].getMax()[0] for i in range(inputList[4].getSize())]
            return (self.addK2ProcSamp(field_E,self.Means[0]),  self.addK2ProcSamp(field_D,self.Means[1]), self.addConst2List(var_Rho,self.Means[2]), self.addConst2List(var_Fpos,self.Means[3]),  self.addConst2List(var_Fnor,self.Means[4]))        
        else :
            field_E = inputList[0].getValues()
            field_D = inputList[1].getValues()
            var_Rho = inputList[2].getValues().getMax()[0] if inputList[2].getValues().getMin()[0]-inputList[2].getValues().getMin()[0]<1e-4 else print('Problem')  
            var_Fpos = inputList[3].getValues().getMax()[0] if inputList[3].getValues().getMin()[0]-inputList[3].getValues().getMin()[0]<1e-4 else print('Problem')  
            var_Fnor = inputList[4].getValues().getMax()[0] if  inputList[4].getValues().getMin()[0]-inputList[4].getValues().getMin()[0]<1e-4 else print('Problem')   
            return field_E+self.Means[0],  field_D+self.Means[1], var_Rho+self.Means[2], var_Fpos+self.Means[3],  var_Fnor+self.Means[4]





    


