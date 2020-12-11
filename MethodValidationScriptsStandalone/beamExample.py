import pickle
from joblib import Parallel, delayed

from numba import jit, int32, float64

import numpy
import openturns as ot
from anastruct.fem.system import SystemElements, Vertex
from anastruct.basic import FEMException
from copy import deepcopy
from scipy.interpolate import CubicSpline
from functools import partial

N = ot.Normal(2,1)
pts_normal_reversed = numpy.arange(2,0,-1*(2/50))
pts_normal = numpy.array(list(map(N.computeCDF, pts_normal_reversed)))
l_0 = pts_normal.sum()
pts_normal = (pts_normal/l_0)
vertices_half = numpy.concatenate([numpy.array([0]),numpy.array(pts_normal*500)])
vertices = numpy.concatenate([vertices_half, vertices_half[::-1]])
for i in range(vertices.shape[0]-1):
    vertices[i+1]=vertices[i+1] + vertices[i]
vertices = numpy.unique(vertices)
vertex_list = [Vertex(vertices[i],0) for i in range(101)]
print('len vertices is:',len(vertex_list))
X = numpy.arange(0,1000+1,10) #to include also 1000
elem_coords = (X[1:]+X[:-1])/2

def experience_mod(young_modulus, diameter, position_force, norm_force,
                   vertices = vertices, vertex_list = vertex_list,
                   elem_coords = elem_coords):
    cs_young_modulus = CubicSpline(elem_coords, young_modulus)
    cs_diameter = CubicSpline(elem_coords, diameter)
    # now we calculate the value of the young modulus and diameter at each element
    young_modu_new = numpy.divide(numpy.add(cs_young_modulus(vertices[1:]),cs_young_modulus(vertices[:-1])), 2)
    diameter_new = numpy.divide(numpy.add(cs_diameter(vertices[1:]),cs_diameter(vertices[:-1])), 2)

    #Here we clip the values to not avec negative young moduli or diameter
    young_modu_new = numpy.clip(a = young_modu_new, a_min = 1000, a_max = None)
    diameter_new = numpy.clip(a = diameter_new, a_min = .1, a_max = None)

    system = SystemElements(EA = None, EI = None) # to make sure we delete the default values
    for k in range(len(vertex_list)-1):      # always a vertex more than element
        system.add_element(location = [vertex_list[k], vertex_list[k+1]],
                            d = diameter_new[k],
                            E = young_modu_new[k],
                            gamma = 7850)
    f_node_id = system.nearest_node('x', position_force)
    system.point_load(f_node_id, Fy = -norm_force)
    system.add_support_hinged(node_id = 1)
    system.add_support_roll(node_id = system.id_last_node , direction='x')

    try :
        solution = system.solve(force_linear = False,
                                verbosity = 50,
                                max_iter = 300,
                                geometrical_non_linear = False,
                                naked = False)
        deflection = numpy.array(system.get_node_result_range('uy'))
        shear = numpy.array(system.get_element_result_range('shear'))
        moment = numpy.array(system.get_element_result_range('moment'))
        element_results = numpy.vstack([young_modu_new, diameter_new, shear, moment])
        norm_position = numpy.array([norm_force, position_force])

    except Exception as e :
        print('Caught exception', e)
        print('------------- PARAMETER LOG -------------')
        print('-- Name : ( mean, variance, min, max)* --')
        print('-- Name : ( value )** -------------------')
        print('-- * : Field , ** : Scalar ')
        print('-- Youngs Modulus : ( {} , {}, {}, {})'.format( round(float(numpy.mean(young_modu_new)), 4),
                                                               round(float(numpy.std(young_modu_new)), 4),
                                                               round(float(numpy.min(young_modu_new)), 4),
                                                               round(float(numpy.max(young_modu_new)), 4)))
        print('-- Diameter : ( {} , {}, {}, {})'.format( round(float(numpy.mean(diameter_new)), 4),
                                                               round(float(numpy.std(diameter_new)), 4),
                                                               round(float(numpy.min(diameter_new)), 4),
                                                               round(float(numpy.max(diameter_new)), 4)))
        print('-- Position Force : ( {} )'.format( round(float(position_force), 4 )))
        print('-- Norm Force : ( {} )'.format( round(float(norm_force), 4 )))

        l_ver = len(vertex_list)
        deflection = numpy.zeros((l_ver,))
        shear = numpy.zeros((l_ver - 1,))
        moment = numpy.zeros((l_ver - 1,))
        element_results = numpy.vstack([young_modu_new, diameter_new, shear, moment])
        norm_position = numpy.array([norm_force, position_force])

    return element_results, deflection, norm_position

@jit(nopython=True)
def nans(shape):
    a = numpy.zeros(shape)
    a.fill(numpy.nan)
    return a

def batchEval(random_young_modulus,
              random_diameter,
              random_forcePos,
              random_forceNorm):
    var1, var2, var3, var4 = random_young_modulus, random_diameter, random_forcePos, random_forceNorm
    result_list = Parallel(n_jobs=-1, verbose=10)(
                    delayed(experience_mod)(
                        var1[i], var2[i], var3[i], var4[i], vertices, vertex_list, elem_coords) for i in range(len(var4)))
    monteCarloResults_elem = numpy.stack(numpy.asarray(result_list)[...,0])
    deflection = numpy.stack(numpy.asarray(result_list)[...,1])
    monteCarloResults_glob = numpy.stack(numpy.asarray(result_list)[...,2])
    print('shape deflection: ', deflection.shape, ' should be [N,10X] something')
    vonMisesStress = getVonMisesStress(monteCarloResults_elem)
    maxDeflection = numpy.amax(numpy.abs(deflection), 1)
    print('deflection std deviation ',numpy.std(maxDeflection))
    return vonMisesStress, maxDeflection

@jit(nopython=True)
def getVonMisesStress(monteCarloResults_elem):
    diameter_MC          = monteCarloResults_elem[:,1,:]
    moment_MC            = monteCarloResults_elem[:,3,:]
    shear_MC             = monteCarloResults_elem[:,2,:]
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


class PureBeam(object):

    def batchEval(self, argList):
        inputs = list(self.convertSinglInputs(argList))
        return batchEval(inputs[0], inputs[1], inputs[2], inputs[3])

    def convertSinglInputs(self, inputList):
        '''To confert the fields into scalars
        '''
        if isinstance(inputList[0],ot.ProcessSample):
            field_E  = numpy.stack([numpy.squeeze(numpy.asarray(inputList[0][i])) for i in range(inputList[0].getSize())])
            field_D  = numpy.stack([numpy.squeeze(numpy.asarray(inputList[1][i])) for i in range(inputList[1].getSize())])
            var_Fpos = numpy.asarray([inputList[2][i][0,0] for i in range(inputList[2].getSize())])
            var_Fnor = numpy.asarray([inputList[3][i][0,0] for i in range(inputList[2].getSize())])
            print('field E shape', field_E.shape)
            print('var_Fnor shape', var_Fnor.shape)
            print('field_E', field_E.mean(), 'field_D', field_D.mean(), 'var_Fpos', var_Fpos.mean(), 'var_Fnor', var_Fnor.mean(),)
            return field_E, field_D, var_Fpos, var_Fnor
        else :
            field_E = inputList[0].getValues()
            field_D = inputList[1].getValues()
            var_Fpos = inputList[2].getValues()[0]
            var_Fnor = inputList[3].getValues()[0]
            return field_E, field_D, var_Fpos, var_Fnor