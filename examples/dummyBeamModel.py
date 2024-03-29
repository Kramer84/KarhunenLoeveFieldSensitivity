from joblib import Parallel, delayed, cpu_count
import numpy
import openturns as ot
from anastruct.fem.system import SystemElements, Vertex
from scipy.interpolate import CubicSpline

try:
    from numba import jit
except:
    print("numba not installed.")
    print("You can install numba if you want to speed up a bit the simulations")

    # Creating dummy function wrapper without jit
    def jit(*args0, **kwargs0):
        def wrapper(func):
            def inner(*args, **kwargs):
                return func(*args, **kwargs)

            return inner

        return wrapper


###############################################################################
# Here we create the list of vertices for our beam, where the density of points is higher in the middle
# We create it here as it is allways used throughout the iterations
# The beam is of length 1M.
pts_normal_reversed = numpy.arange(2, 0, -1 * (2 / 50))
pts_normal = numpy.array(list(map(ot.Normal(2, 1).computeCDF, pts_normal_reversed)))
l_0 = pts_normal.sum()
pts_normal = pts_normal / l_0
vertices_half = numpy.concatenate([numpy.array([0]), numpy.array(pts_normal * 500)])
VERTICES = numpy.concatenate([vertices_half, vertices_half[::-1]])

for i in range(VERTICES.shape[0] - 1):
    VERTICES[i + 1] = VERTICES[i + 1] + VERTICES[i]

VERTICES = numpy.unique(VERTICES)
VERTEX_LIST = [Vertex(VERTICES[i], 0) for i in range(101)]
N_VERTEX = len(VERTEX_LIST)
print("len vertices is:", N_VERTEX)
X = numpy.arange(0, 1000 + 1, 10)  # to include also 1000
ELEM_COORDS = (X[1:] + X[:-1]) / 2
###############################################################################


def experience_mod(young_modulus, diameter, position_force, norm_force):
    cs_young_modulus = CubicSpline(ELEM_COORDS, young_modulus)
    cs_diameter = CubicSpline(ELEM_COORDS, diameter)
    # now we calculate the value of the young modulus and diameter at each element
    young_modu_new = numpy.divide(
        numpy.add(cs_young_modulus(VERTICES[1:]), cs_young_modulus(VERTICES[:-1])), 2.0
    )
    diameter_new = numpy.divide(
        numpy.add(cs_diameter(VERTICES[1:]), cs_diameter(VERTICES[:-1])), 2.0
    )

    system = SystemElements(
        EA=None, EI=None
    )  # to make sure we delete the default values
    for k in range(len(VERTEX_LIST) - 1):  # always a vertex more than element
        system.add_element(
            location=[VERTEX_LIST[k], VERTEX_LIST[k + 1]],
            d=diameter_new[k],
            E=young_modu_new[k],
            gamma=7850,
        )  # density of steel
    f_node_id = system.nearest_node("x", position_force)
    system.point_load(f_node_id, Fy=-norm_force)
    system.add_support_hinged(node_id=1)
    system.add_support_roll(node_id=system.id_last_node, direction="x")

    try:
        solution = system.solve(
            force_linear=False,
            verbosity=50,
            max_iter=300,
            geometrical_non_linear=False,
            naked=False,
        )
        deflection = numpy.array(system.get_node_result_range("uy"))
        shear = numpy.array(system.get_element_result_range("shear"))
        moment = numpy.array(system.get_element_result_range("moment"))
        element_results = numpy.vstack([young_modu_new, diameter_new, shear, moment])
        norm_position = numpy.array([norm_force, position_force])

    except Exception as e:
        print("Caught exception", e)
        print("------------- PARAMETER LOG -------------")
        print("-- Name : ( mean, variance, min, max)* --")
        print("-- Name : ( value )** -------------------")
        print("-- * : Field , ** : Scalar ")
        print(
            "-- Youngs Modulus : ( {} , {}, {}, {})".format(
                round(float(numpy.mean(young_modu_new)), 4),
                round(float(numpy.std(young_modu_new)), 4),
                round(float(numpy.min(young_modu_new)), 4),
                round(float(numpy.max(young_modu_new)), 4),
            )
        )
        print(
            "-- Diameter : ( {} , {}, {}, {})".format(
                round(float(numpy.mean(diameter_new)), 4),
                round(float(numpy.std(diameter_new)), 4),
                round(float(numpy.min(diameter_new)), 4),
                round(float(numpy.max(diameter_new)), 4),
            )
        )
        print("-- Position Force : ( {} )".format(round(float(position_force), 4)))
        print("-- Norm Force : ( {} )".format(round(float(norm_force), 4)))

        deflection = numpy.zeros((N_VERTEX,))
        shear = numpy.zeros((N_VERTEX - 1,))
        moment = numpy.zeros((N_VERTEX - 1,))
        element_results = numpy.vstack([young_modu_new, diameter_new, shear, moment])
        norm_position = numpy.array([norm_force, position_force])

    return element_results, deflection, norm_position


@jit(nopython=True)
def nans(shape):
    a = numpy.zeros(shape)
    a.fill(numpy.nan)
    return a


def batchEval(random_young_modulus, random_diameter, random_forcePos, random_forceNorm):
    var1, var2, var3, var4 = (
        random_young_modulus,
        random_diameter,
        random_forcePos,
        random_forceNorm,
    )
    result_list = Parallel(n_jobs=cpu_count(), verbose=10)(
        delayed(experience_mod)(var1[i], var2[i], var3[i], var4[i])
        for i in range(len(var4))
    )
    monteCarloResults_elem = numpy.stack(numpy.asarray(result_list)[..., 0])
    deflection = numpy.stack(numpy.asarray(result_list)[..., 1])
    monteCarloResults_glob = numpy.stack(numpy.asarray(result_list)[..., 2])
    print("shape deflection: ", deflection.shape, " should be [N,10X] something")
    vonMisesStress = getVonMisesStress(monteCarloResults_elem)
    maxDeflection = numpy.amax(numpy.abs(deflection), 1)
    print("deflection std deviation ", numpy.std(maxDeflection))
    return vonMisesStress, maxDeflection


@jit(nopython=True)
def getVonMisesStress(monteCarloResults_elem):
    diameter_MC = monteCarloResults_elem[:, 1, :]
    moment_MC = monteCarloResults_elem[:, 3, :]
    shear_MC = monteCarloResults_elem[:, 2, :]
    inertia_MC, area_MC = moment_inertia_PlainRoundBeam(diameter_MC)
    maxbendingStress_MC = getMaximumBendingStress(moment_MC, inertia_MC, diameter_MC)
    shearStress_MC = (4 / 3) * numpy.divide(shear_MC, area_MC)
    vonMisesCriteria = numpy.sqrt(
        numpy.square(maxbendingStress_MC)
        + numpy.multiply(numpy.square(shearStress_MC), 3)
    )
    return vonMisesCriteria


@jit(nopython=True)
def moment_inertia_PlainRoundBeam(D):
    # Function calculating the moment of inertia as well as the area of a beam given a diameter
    return numpy.divide(numpy.pi * numpy.power(D, 4), 64), numpy.pi * numpy.power(
        numpy.divide(D, 2), 2
    )


@jit(nopython=True)
def getMaximumBendingStress(bendingMoment, inertia, diameter):
    return numpy.multiply(numpy.divide(bendingMoment, inertia), -1 * diameter / 2)


class Beam(object):
    """Class representing a beam supported on both ends and where a force
    is applied in the middle. Stochastic uncertainties are on the material
    properties and the diameter, and scalar uncertainties are on the point
    of application of the force, and on the norm of the force.
    """

    def batchEval(self, argList):
        inputs = list(self.convertSingleInputs(argList))
        return batchEval(inputs[0], inputs[1], inputs[2], inputs[3])

    def convertSingleInputs(self, inputList):
        """To confert the fields into scalars"""
        if isinstance(inputList[0], ot.ProcessSample):
            field_E = numpy.stack(
                [
                    numpy.squeeze(numpy.asarray(inputList[0][i]))
                    for i in range(inputList[0].getSize())
                ]
            )
            field_D = numpy.stack(
                [
                    numpy.squeeze(numpy.asarray(inputList[1][i]))
                    for i in range(inputList[1].getSize())
                ]
            )
            var_Fpos = numpy.asarray(
                [inputList[2][i][0, 0] for i in range(inputList[2].getSize())]
            )
            var_Fnor = numpy.asarray(
                [inputList[3][i][0, 0] for i in range(inputList[2].getSize())]
            )
            print("field E shape", field_E.shape)
            print("var_Fnor shape", var_Fnor.shape)
            print(
                "field_E",
                field_E.mean(),
                "field_D",
                field_D.mean(),
                "var_Fpos",
                var_Fpos.mean(),
                "var_Fnor",
                var_Fnor.mean(),
            )
            return field_E, field_D, var_Fpos, var_Fnor
        else:
            field_E = inputList[0].getValues()
            field_D = inputList[1].getValues()
            var_Fpos = inputList[2].getValues()[0]
            var_Fnor = inputList[3].getValues()[0]
            return field_E, field_D, var_Fpos, var_Fnor
