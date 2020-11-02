import pickle
from joblib import Parallel, delayed

from numba import jit, int32, float64

import numpy
import openturns as ot
from anastruct.fem.system import SystemElements, Vertex
from copy import deepcopy

class _BeamBase(object):

    def __init__(self, mesh):
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
                        delayed(experience)(
                            var1[i], var2[i], var3[i], var4[i], var5[i], var6, var7, i) for i in range(len(var5)))
        monteCarloResults_elem = numpy.stack(numpy.asarray(result_list)[...,0])
        monteCarloResults_node = numpy.stack(numpy.asarray(result_list)[...,1])
        monteCarloResults_glob = numpy.stack(numpy.asarray(result_list)[...,2])
        deflection = monteCarloResults_node[:,1,:]
        print('shape deflection: ', deflection.shape, ' should be [N,10X] something')
        vonMisesStress = self.getVonMisesStress(monteCarloResults_elem)
        maxDeflection = numpy.amax(numpy.abs(deflection), 1)
        print('deflection std deviation ',numpy.std(maxDeflection))
        return vonMisesStress, maxDeflection

    def singleEval(self ,random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm):
        vertsList = self.vertices
        lElem = self.lElem
        result = experience(numpy.squeeze(numpy.array(random_young_modulus)), 
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
        inertia, area  = moment_inertia_PlainRoundBeam(diameter)
        maxbendingStress  = getMaximumBendingStress(moment, inertia, diameter)
        shearStress   = (4/3)*numpy.divide(shear, area)  
        vonMisesCriteria = numpy.sqrt(numpy.square(maxbendingStress) + numpy.multiply(numpy.square(shearStress), 3))
        maxDeflection = numpy.max(numpy.abs(node_results[1,...]))
        return  vonMisesCriteria, maxDeflection

    def getVonMisesStress(self,  monteCarloResults_elem=None):
        assert monteCarloResults_elem is not None
        diameter_MC          = monteCarloResults_elem[:,2,:]
        moment_MC            = monteCarloResults_elem[:,4,:]
        shear_MC             = monteCarloResults_elem[:,3,:]
        inertia_MC, area_MC  = moment_inertia_PlainRoundBeam(diameter_MC)
        maxbendingStress_MC  = getMaximumBendingStress(moment_MC, inertia_MC, diameter_MC)
        shearStress_MC       = (4/3)*numpy.divide(shear_MC, area_MC)  
        vonMisesCriteria     = numpy.sqrt(numpy.square(maxbendingStress_MC) + numpy.multiply(numpy.square(shearStress_MC), 3))
        return vonMisesCriteria

@jit(nopython=True)
def moment_inertia_PlainRoundBeam(D):
    # Function calculating the moment of inertia as well as the area of a beam given a diameter
    return numpy.divide(numpy.pi*numpy.power(D,4), 64), numpy.pi*numpy.power(numpy.divide(D,2),2)

@jit(nopython=True)
def getMaximumBendingStress(bendingMoment, inertia, diameter):
    return numpy.multiply(numpy.divide(bendingMoment, inertia),-1*diameter/2)

### For multiprocessing purposes
    
    # On enleve le self de toutes les fonctions utilisées en multiprocessing, et en pase en méthode statique
    # car comme on posse en argument de la classe un objet NdGaussianProcessConstructor, le "self" ne peut plus
    # etre pickle, du coup en multiprocessing ca nous envoie une erreur, comme quoi un objet ne peut pas etre déserialisé
    # ce qui dans notre cas devrait être le "self" contenant une autre classe

def experience( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element, i=0):
    '''Function that is used for multiprocessing the beam experience
    There are a lot of tests to make sure the codes don't crash
    '''
    youngModu     = youngModu
    diam          = diam
    n0_elems      = diam.shape[0]
    density       = density
    forcePosition = forcePosition
    forceNorm     = forceNorm
    if i % 33 == 0 :
        print('Iteration',i)
        print('youngModu.shape is', youngModu.shape)
        print('diam.shape is', diam.shape)
        print('n0_elems is', n0_elems)
        print('density is', density)
        print('forcePosition is', forcePosition)
        print('forceNorm is', forceNorm)
    
        print('youngModu.mean() is', youngModu.mean())
        print('diam.mean() is', diam.mean())

    BeamObj_MP, youngModu, diam = anastruct_beam_MP(youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element)
    try :
        solution          = BeamObj_MP.solve(force_linear = False,
                                             verbosity = 50,
                                             max_iter = 300, 
                                             geometrical_non_linear = False,
                                             naked = False)
        points_range      = numpy.array(BeamObj_MP.nodes_range('x'), copy=False, subok = True)
        elem_length_range = points_range[1:]-(points_range[1]-points_range[0])/2
        deflection        = numpy.array(BeamObj_MP.get_node_result_range('uy'), copy=False, subok = True)
        shear             = numpy.array(BeamObj_MP.get_element_result_range('shear'), copy=False, subok = True)
        moment            = numpy.array(BeamObj_MP.get_element_result_range('moment'), copy=False, subok = True)
        element_results   = numpy.vstack([elem_length_range,youngModu, diam, 
                                          numpy.array(shear, copy=False, subok = True), 
                                          numpy.array(moment, copy=False, subok = True)])
        node_results      = numpy.vstack([points_range, deflection])
        global_beamParams = numpy.array([forceNorm, density, forcePosition], copy=False, subok = True)

        return element_results, node_results, global_beamParams

    except :
        print('there was an error in the stiffness matrix.\n','Filling  with numpy.nan values\n')
        elem_length_range   = youngModu  = diam =  shear =  moment = nans(shape=(n0_elems+1,))
        points_range        = deflection = nans(shape=(n0_elems+2,))
        element_results     = numpy.vstack([elem_length_range,youngModu, diam, shear, moment])
        node_results        = numpy.vstack([points_range, deflection])
        global_beamParams   = nans(shape=(3,))
        return element_results, node_results, global_beamParams

def anastruct_beam_MP( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element):
    system = SystemElements(EA = None, EI = None) # to make sure we delete the default values
    for k in range(len(vertex_list)-1):      # always a vertex more than element
        system.add_element(location = [vertex_list[k], vertex_list[k+1]], 
                            d = diam[k],
                            E = youngModu[k],
                            gamma = density)
    
    nodeID, random_diameter, random_young_modulus = nodeChecksAndVertsInsertion_MP(system, forcePosition, diam, youngModu, l_element)
    system.point_load(nodeID, Fy = -forceNorm)   # in kN
    system.add_support_hinged(node_id=min(system.node_map.keys()))
    system.add_support_roll(node_id=system.id_last_node , direction='x')
    return system, random_young_modulus, random_diameter

    
def nodeChecksAndVertsInsertion_MP( system, forcePosition, random_diameter, random_young_modulus, l_element):
    # Function to apply force at force position
    nearestNodeFposID = system.nearest_node('x', forcePosition)
    elemNumber, factor = getElemIdAndFactor_MP(forcePosition, l_element)

    if factor*l_element<1: # if the application point is to near wuth an ither point
        ## We add an other node next to it so we normalize the number of elements
        system.insert_node(element_id = nearestNodeFposID+2, factor=0.5)
        random_diameter      = insertToPos(random_diameter, nearestNodeFposID+2)
        random_young_modulus = insertToPos(random_young_modulus, nearestNodeFposID+2)
        return nearestNodeFposID, random_diameter, random_young_modulus

    elif forcePosition < l_element*.1 :
        system.insert_node(element_id = 1, factor= 0.1)
        random_diameter = insertToPos(random_diameter, 0)
        random_young_modulus = insertToPos(random_young_modulus, 0)
        return 0, random_diameter, random_young_modulus

    else :
        system.insert_node(element_id = elemNumber, factor=factor)
        # as we inserted a node we have one more element so we add it 
        random_diameter      = insertToPos(random_diameter, nearestNodeFposID-2)
        random_young_modulus = insertToPos(random_young_modulus, nearestNodeFposID-2)
        return nearestNodeFposID, random_diameter, random_young_modulus


######################################################################################################
######################################################################################################
######################################################################################################


def experience_mod( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element, i=0):
    '''Function that is used for multiprocessing the beam experience
    There are a lot of tests to make sure the codes don't crash
    '''
    youngModu     = youngModu
    diam          = diam
    n0_elems      = diam.shape[0]
    density       = density
    forcePosition = forcePosition
    forceNorm     = forceNorm
    if i % 33 == 0 :
        print('Iteration',i)
        print('youngModu.shape is', youngModu.shape)
        print('diam.shape is', diam.shape)
        print('n0_elems is', n0_elems)
        print('density is', density)
        print('forcePosition is', forcePosition)
        print('forceNorm is', forceNorm)
    
        print('youngModu.mean() is', youngModu.mean())
        print('diam.mean() is', diam.mean())

    BeamObj_MP, youngModu, diam = anastruct_beam_MP(youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element)
    try :
        solution          = BeamObj_MP.solve(force_linear = False,
                                             verbosity = 50,
                                             max_iter = 300, 
                                             geometrical_non_linear = False,
                                             naked = False)
        points_range      = numpy.array(BeamObj_MP.nodes_range('x'), copy=False, subok = True)
        elem_length_range = points_range[1:]-(points_range[1]-points_range[0])/2
        deflection        = numpy.array(BeamObj_MP.get_node_result_range('uy'), copy=False, subok = True)
        shear             = numpy.array(BeamObj_MP.get_element_result_range('shear'), copy=False, subok = True)
        moment            = numpy.array(BeamObj_MP.get_element_result_range('moment'), copy=False, subok = True)
        element_results   = numpy.vstack([elem_length_range,youngModu, diam, 
                                          numpy.array(shear, copy=False, subok = True), 
                                          numpy.array(moment, copy=False, subok = True)])
        node_results      = numpy.vstack([points_range, deflection])
        global_beamParams = numpy.array([forceNorm, density, forcePosition], copy=False, subok = True)

        return element_results, node_results, global_beamParams

    except :
        print('there was an error in the stiffness matrix.\n','Filling  with numpy.nan values\n')
        elem_length_range   = youngModu  = diam =  shear =  moment = nans(shape=(n0_elems+1,))
        points_range        = deflection = nans(shape=(n0_elems+2,))
        element_results     = numpy.vstack([elem_length_range,youngModu, diam, shear, moment])
        node_results        = numpy.vstack([points_range, deflection])
        global_beamParams   = nans(shape=(3,))
        return element_results, node_results, global_beamParams

def anastruct_beam_MP( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element):
    system = SystemElements(EA = None, EI = None) # to make sure we delete the default values
    for k in range(len(vertex_list)-1):      # always a vertex more than element
        system.add_element(location = [vertex_list[k], vertex_list[k+1]], 
                            d = diam[k],
                            E = youngModu[k],
                            gamma = density)
    
    nodeID, random_diameter, random_young_modulus = nodeChecksAndVertsInsertion_MP(system, forcePosition, diam, youngModu, l_element)
    system.point_load(nodeID, Fy = -forceNorm)   # in kN
    system.add_support_hinged(node_id=min(system.node_map.keys()))
    system.add_support_roll(node_id=system.id_last_node , direction='x')
    return system, random_young_modulus, random_diameter

    
def nodeChecksAndVertsInsertion_MP( system, forcePosition, random_diameter, random_young_modulus, l_element):
    # Function to apply force at force position
    nearestNodeFposID = system.nearest_node('x', forcePosition)
    elemNumber, factor = getElemIdAndFactor_MP(forcePosition, l_element)

    if factor*l_element<1: # if the application point is to near wuth an ither point
        ## We add an other node next to it so we normalize the number of elements
        system.insert_node(element_id = nearestNodeFposID+2, factor=0.5)
        random_diameter      = insertToPos(random_diameter, nearestNodeFposID+2)
        random_young_modulus = insertToPos(random_young_modulus, nearestNodeFposID+2)
        return nearestNodeFposID, random_diameter, random_young_modulus

    elif forcePosition < l_element*.1 :
        system.insert_node(element_id = 1, factor= 0.1)
        random_diameter = insertToPos(random_diameter, 0)
        random_young_modulus = insertToPos(random_young_modulus, 0)
        return 0, random_diameter, random_young_modulus

    else :
        system.insert_node(element_id = elemNumber, factor=factor)
        # as we inserted a node we have one more element so we add it 
        random_diameter      = insertToPos(random_diameter, nearestNodeFposID-2)
        random_young_modulus = insertToPos(random_young_modulus, nearestNodeFposID-2)
        return nearestNodeFposID, random_diameter, random_young_modulus


######################################################################################################
######################################################################################################
######################################################################################################



@jit(nopython=True)
def insertToPos(array, pos):
    if pos > 0:
        array = numpy.concatenate((array[:pos], numpy.asarray([array[pos]]), array[pos:]))
        return array
    else : 
        array = numpy.concatenate((numpy.asarray([array[0]]), array))
        return array

@jit(nopython=True)
def getElemIdAndFactor_MP( forcePosition, l_element):
    idxElem = int(forcePosition // l_element)
    fraction = (forcePosition - idxElem*l_element)/l_element
    return idxElem, fraction # +1 because anastruct counts elements from 1 

@jit(nopython=True)
def nans(shape):
    a = numpy.zeros(shape)
    a.fill(numpy.nan)
    return a



class PureBeam(object):

    def __init__(self, mesh):
        self.Beam = _BeamBase(mesh)
        self.vertsList = self.Beam.vertices
        self.lElem = self.Beam.lElem

    def singleEval(self, argList):
        inputs = list(self.convertSinglInputs(argList))
        inputs.extend([self.vertsList, self.lElem])
        return self.Beam.singleEval(*inputs)

    def batchEval(self, argList):
        inputs = list(self.convertSinglInputs(argList))
        return self.Beam.batchEval(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]) 
            
    def convertSinglInputs(self, inputList):
        '''To confert the fields into scalars
        '''
        if isinstance(inputList[0],ot.ProcessSample):
            field_E  = numpy.stack([numpy.squeeze(numpy.asarray(inputList[0][i])) for i in range(inputList[0].getSize())])
            field_D  = numpy.stack([numpy.squeeze(numpy.asarray(inputList[1][i])) for i in range(inputList[1].getSize())])
            var_Rho  = numpy.asarray([inputList[2][i][0,0] for i in range(inputList[2].getSize())])
            var_Fpos = numpy.asarray([inputList[3][i][0,0] for i in range(inputList[2].getSize())]) 
            var_Fnor = numpy.asarray([inputList[4][i][0,0] for i in range(inputList[2].getSize())])
            print('field E shape', field_E.shape)
            print('var_Fnor shape', var_Fnor.shape)
            print('field_E', field_E.mean(), 'field_D', field_D.mean(), 'var_Rho', var_Rho.mean(), 'var_Fpos', var_Fpos.mean(), 'var_Fnor', var_Fnor.mean(),)
            return field_E, field_D, var_Rho, var_Fpos, var_Fnor
        else :
            field_E = inputList[0].getValues()
            field_D = inputList[1].getValues()
            var_Rho = inputList[2].getValues()[0]
            var_Fpos = inputList[3].getValues()[0]
            var_Fnor = inputList[4].getValues()[0]
            return field_E, field_D, var_Rho, var_Fpos, var_Fnor


#Here we create half of the coordinates of the beam
#The length decreases inversly proportional to the Normal law
N = ot.Normal(2,1)
pts_normal_reversed = np.arange(2,0,-1*(2/50))
pts_normal = np.array(list(map(N.computeCDF, pts_normal_reversed)))
l_0 = pts_normal.sum()
pts_normal/l_0



N.computeCDF(ot.Sample(np.expand_dims(pts_normal_reversed,axis=1)))

def buildSystemParameters(young_modulus, diameter, density, 
                position_force, norm_force,length_beam=1000, i=0):
    n_elems = young_modulus.shape[0]
    l = length_beam
    n_verts = n_elems + 1
    l_elem = length_beam/n_elems
    l_sub_elem = l_elem/10
 