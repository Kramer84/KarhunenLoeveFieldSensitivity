__version__ = '0.1'
__author__ = 'Kristof Attila S.'
__date__  = '15.03.20'

import  io
import  os
import  pickle
from    joblib               import  Parallel, delayed, cpu_count

from    PIL                  import  Image
import  numpy
import  openturns
import  matplotlib.pyplot    as      pyplot
from    anastruct.fem.system import  SystemElements, Vertex

'''analysis of model in tutorial : notebook/Demo_Analyse_Sensibilite_poutre.ipynb
'''

class RandomBeam_anastruct(object):
    '''Class to create a finite element beam where each parameter is random 

    The beam in this class has each one of it's elements diameter and young 
    modulus set by a stochastic process. The force and position of the force
    can also be modelled by a random variable.
    The difficulty is that the process gorverning the beam is defined in advance
    with a precise number of elements. And that if that number of elements is 
    not a multiple of 2, we have to add a vertex in the middle to apply the force;
    The effect of this is that in our model we have one more vertex and element,
    thus we can not directly apply the results of our processes. 

    Note
    ----
    Most of the logic behind the definition of the elements is taken from 
    Ritchie Vink's blog : 
    https://www.ritchievink.com/blog/2017/08/23/a-nonlinear-water-accumulation-analysis-in-python/
    Ritchie Vink is the author of anastruct.
    '''

    def __init__(self, process_E     = None, 
                       process_D     = None, 
                       process_Rho   = None,      ## rho en kg/m³
                       process_Fpos  = None,
                       process_Fnorm = None):   ## fpos en mm
    
        assert (process_E.getDimension() == 1 and process_D.getMesh() \
                == process_E.getMesh()) , "Only pass one dimensional gaussian process"
        self.process_young           = process_E     # MPa  Stochastic Process
        self.process_diam            = process_D     # mm   Stochastic Process
        self.process_Rho             = process_Rho   # to later divide by 100 to be in kN/m³ Normal Distribution
        self.process_Fpos            = process_Fpos  # m    Normal Distribution
        self.process_Fnorm           = process_Fnorm # N    Normal Distribution
        self.mesh                    = self.process_diam.getMesh() #it is a regular mesh
        self.l_element               = (self.mesh.getUpperBound() \
                            - self.mesh.getLowerBound()).norm()/self.mesh.getSimplicesNumber()
        self.vertex_list             = self._makeVerticesList()
        self.BeamObj                 = None
        self.random_young_modulus    = None
        self.random_diameter         = None
        self.random_density          = None 
        self.random_forcePos         = None
        self.random_forceNorm        = None 
        self.monteCarloResults_elem  = None
        self.monteCarloResults_node  = None
        self.monteCarloResults_glob  = None
        self.modelsWithErrors        = list() 

    def solveProblem(self, returnSolution = False):
        solution = self.BeamObj.solve(force_linear=False, max_iter=400, geometrical_non_linear = False)
        if returnSolution == True:
            return solution

    def _makeVerticesList(self):
        n_elems     = self.mesh.getVerticesNumber()
        vertices    = numpy.arange(n_elems+1)*((self.mesh.getUpperBound() - self.mesh.getLowerBound()).norm()/(n_elems))
        vertex_list = [Vertex(vertices[i],0) for i in range(n_elems+1)]
        return vertex_list

    def getRandomBeamParameters(self):
        self.random_young_modulus       = self.process_young.getRealization(getAsArray = True)
        self.random_diameter            = self.process_diam.getRealization(getAsArray = True)
        self.random_density             = self.process_Rho.getRealization()
        self.random_forcePos            = self.process_Fpos.getRealization()
        self.random_forceNorm           = self.process_Fnorm.getRealization()
        return self.random_young_modulus, self.random_diameter, self.random_density, self.random_forcePos, self.random_forceNorm

    def instantiateRandomBeam(self):
        youngModu, diam, density, forcePosition, forceNorm = self.getRandomBeamParameters()
        self.anastruct_beam(youngModu, diam, density, forcePosition, forceNorm)

    def anastruct_beam(self, youngModu, diam, density, forcePosition, forceNorm):
        system = SystemElements(EA = None, EI = None) # to make sure we delete the default values
        for k in range(len(self.vertex_list)-1):      # always a vertex more than element
            system.add_element(location = [self.vertex_list[k], self.vertex_list[k+1]], 
                                d       = float(diam[k]),
                                E       = float(youngModu[k]),
                                gamma   = density)
        
        nodeID, system = self.nodeChecksAndVertsInsertion(system, forcePosition)
        system.point_load(nodeID, Fy = -forceNorm)   # in kN
        system.add_support_hinged(node_id=1)
        system.add_support_roll(node_id=system.id_last_node , direction='x')
        self.BeamObj = system

    def nodeChecksAndVertsInsertion(self, system, forcePosition):
        # Function to apply force at force position
        nearestNodeFposID = system.nearest_node('x', forcePosition)
        if abs(self.vertex_list[nearestNodeFposID].x - forcePosition)<1.:
            nodeID_F = nearestNodeFposID
            ## We add an other node next to it so we normalize the number of elements
            system.insert_node(element_id = nodeID_F+1, factor=0.5)
            self.random_diameter      = numpy.insert(self.random_diameter, nodeID_F-1, self.random_diameter[nodeID_F-1])
            self.random_young_modulus = numpy.insert(self.random_young_modulus, nodeID_F-1, self.random_young_modulus[nodeID_F-1])
            return nodeID_F, system 
        else :
            elemNumber, factor = self.getElemIdAndFactor(forcePosition)
            system.insert_node(element_id = elemNumber, factor=factor)
            newNodeID = system.nearest_node('x',forcePosition)
            
            # as we inserted a node we have one more element so we add it 
            self.random_diameter      = numpy.insert(self.random_diameter, newNodeID-2, self.random_diameter[newNodeID-2])
            self.random_young_modulus = numpy.insert(self.random_young_modulus, newNodeID-2, self.random_young_modulus[newNodeID-2])
            return newNodeID, system

    def getElemIdAndFactor(self, forcePosition):
        elemNumber = int(forcePosition/self.l_element)
        factor     = forcePosition/self.l_element - elemNumber
        #print('forcePosition= ',forcePosition,'elemNumber= ',elemNumber, 'factor= ',factor)
        return elemNumber+1, factor # +1 because anastruct counts elements from 1 
 
    def moment_inertia_PlainRoundBeam(self, D):
        # Function calculating the moment of inertia as well as the area of a beam given a diameter
        return numpy.divide(numpy.pi*numpy.power(D,4), 64), numpy.pi*numpy.power(numpy.divide(D,2),2)

    def getMaximumBendingStress(self, bendingMoment, inertia, diameter):
        assert bendingMoment.shape == inertia.shape , "same shape required"
        return numpy.multiply(numpy.divide(bendingMoment, inertia),-1*diameter/2)

    def postprecess_beam(self, returnStuff = False):
        # Function to get the different values of the simulation and do some plotting. This is not for the monte carlo simulation
        deflection  = numpy.asarray(self.BeamObj.get_node_result_range('uy'))
        shear       = numpy.asarray(self.BeamObj.get_element_result_range('shear'))
        moment      = numpy.asarray(self.BeamObj.get_element_result_range('moment'))
        axial       = numpy.asarray(self.BeamObj.get_element_result_range('axial'))
        tempList = [deflection, shear, moment]
        if returnStuff == True:
            return tempList

#####################################################################################################
#####################################################################################################
#####################################################################################################
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
        youngModu     = numpy.squeeze(youngModu).astype(float)
        diam          = numpy.squeeze(diam).astype(float)
        n0_elems      = len(diam)
        density       = float(density)
        forcePosition = float(forcePosition)
        forceNorm     = float(forceNorm)
        BeamObj_MP, youngModu, diam, density, forcePosition, forceNorm = RandomBeam_anastruct.instantiateRandomBeam_MP(youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element)
        try :
            solution                  = BeamObj_MP.solve(force_linear           = False,
                                                         verbosity              = 0,
                                                         max_iter               = 300, 
                                                         geometrical_non_linear = False,
                                                         naked                  = False)
            points_range              = numpy.asarray(BeamObj_MP.nodes_range('x'))
            elem_length_range         = points_range[1:]-(points_range[1]-points_range[0])/2
            deflection, shear, moment = RandomBeam_anastruct.postprecess_beam_MP(BeamObj_MP)
            element_results           = numpy.vstack([elem_length_range,youngModu, diam, numpy.asarray(shear), numpy.asarray(moment)])
            node_results              = numpy.vstack([points_range, deflection])
            global_beamParams         = numpy.array([forceNorm, density, forcePosition])
            return element_results, node_results, global_beamParams

        except Exception as e:
            print('there was an error in the stiffness matrix.\n','Error =',str(e),'\n filling gaps with numpy.nan values\n\n')
            #self.modelsWithErrors.append(BeamObj_MP)
            points_range        = numpy.asarray(BeamObj_MP.nodes_range('x'))
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
        a, b, c=RandomBeam_anastruct.anastruct_beam_MP(youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element)
        return a, b, c, density, forcePosition, forceNorm

    @staticmethod
    def anastruct_beam_MP( youngModu, diam, density, forcePosition, forceNorm, vertex_list, l_element):
        system = SystemElements(EA = None, EI = None) # to make sure we delete the default values
        for k in range(len(vertex_list)-1):      # always a vertex more than element
            system.add_element(location = [vertex_list[k], vertex_list[k+1]], 
                                d       = float(diam[k]),
                                E       = float(youngModu[k]),
                                gamma   = density)
        
        nodeID, system, random_diameter, random_young_modulus = RandomBeam_anastruct.nodeChecksAndVertsInsertion_MP(system, forcePosition, diam, youngModu, vertex_list, l_element)
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
            elemNumber, factor = RandomBeam_anastruct.getElemIdAndFactor_MP(forcePosition, l_element)
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
        return [numpy.asarray(BeamObj.get_node_result_range('uy')), 
                numpy.asarray(BeamObj.get_element_result_range('shear')), 
                numpy.asarray(BeamObj.get_element_result_range('moment'))]

#####################################################################################################
#####################################################################################################
#####################################################################################################

    def monteCarlo_experience(self, n_simu = 10000):
        self.getRandomBeamParameters_MP(n_simu)
        random_young_modulus = self.process_young.sample_map
        random_diameter      = self.process_diam.sample_map
        random_density       = self.process_Rho.sample
        random_forcePos      = self.process_Fpos.sample
        random_forceNorm     = self.process_Fnorm.sample
        monteCarloResults_elem, monteCarloResults_node, monteCarloResults_glob = self.multiprocessBatchField(random_young_modulus.tolist(), 
                                                                                                             random_diameter.tolist(),
                                                                                                             random_density.tolist(),
                                                                                                             random_forcePos.tolist(),
                                                                                                             random_forceNorm.tolist())
        self.monteCarloResults_elem = monteCarloResults_elem
        self.monteCarloResults_node = monteCarloResults_node
        self.monteCarloResults_glob = monteCarloResults_glob

    def _execute_sample(self, X):
        field_young, field_diam, density, forcePos, forceNorm = self.buildArgsFromCollectionOfNormals(X)
        monteCarloResults_elem, monteCarloResults_node, monteCarloResults_glob = self.multiprocessBatchField(field_young, 
                                                                                                             field_diam, 
                                                                                                             density, 
                                                                                                             forcePos, 
                                                                                                             forceNorm)
        vonMisesStress = self.getVonMisesStress(monteCarloResults_elem)
        return vonMisesStress

    def _execute(self, X):
        field_young, field_diam, density, forcePos, forceNorm = self.buildArgsFromCollectionOfNormals(X)
        self.anastruct_beam(numpy.squeeze(numpy.asarray(field_young)), numpy.squeeze(numpy.asarray(field_diam)), density, forcePos, forceNorm)
        self.solveProblem()
        deflection, shear, moment = self.postprecess_beam(returnStuff = True)
        inertia, area             = self.moment_inertia_PlainRoundBeam(numpy.squeeze(numpy.asarray(field_diam)))
        maxbendingStress          = self.getMaximumBendingStress(moment, inertia, numpy.squeeze(numpy.asarray(field_diam)))
        shearStress               = (4/3)*numpy.divide(shear, area)  
        vonMisesCriteria          = numpy.sqrt(numpy.square(maxbendingStress) + numpy.multiply(numpy.square(shearStress), 3))
        return vonMisesCriteria

    def buildArgsFromCollectionOfNormals(self, collectionOfRealizations):
        n_elemsKL_young = self.process_young.decompositionAsRandomVector.n_modes
        n_elemsKL_diam  = self.process_diam.decompositionAsRandomVector.n_modes
        arrayOfReals    = numpy.asarray(collectionOfRealizations)
        elemsYoung      = arrayOfReals[..., : n_elemsKL_young]
        elemsDiam       = arrayOfReals[..., n_elemsKL_young : n_elemsKL_young+n_elemsKL_diam]
        assert arrayOfReals[...,n_elemsKL_young+n_elemsKL_diam:].shape[-1] == 3 ,"only three random variables"
        field_young = self.process_young.liftDistributionToField(elemsYoung.tolist())
        field_diam  = self.process_diam.liftDistributionToField(elemsDiam.tolist())
        otherVars   = arrayOfReals[...,n_elemsKL_young+n_elemsKL_diam:]
        return field_young, field_diam, otherVars[...,0].tolist(), otherVars[...,1].tolist(), otherVars[...,2].tolist()

    def multiprocessBatchField(self, random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm):
        var1, var2, var3, var4, var5, var6, var7 = random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm, self.vertex_list, self.l_element
        result_list = Parallel(n_jobs=-1, verbose=10)(
                        delayed(RandomBeam_anastruct.experience)(
                            var1[i], var2[i], var3[i], var4[i], var5[i], var6, var7) for i in range(len(var5)))
        monteCarloResults_elem = numpy.stack(numpy.asarray(result_list)[...,0])
        monteCarloResults_node = numpy.stack(numpy.asarray(result_list)[...,1])
        monteCarloResults_glob = numpy.stack(numpy.asarray(result_list)[...,2])
        return monteCarloResults_elem, monteCarloResults_node, monteCarloResults_glob
        
    def experienceBeam(self, random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm):
        var1, var2, var3, var4, var5, var6, var7 = random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm, self.vertex_list, self.l_element
        result = RandomBeam_anastruct.experience(var1, var2, var3, var4, var5, var6, var7) 
        monteCarloResults_elem = numpy.stack(numpy.asarray(result)[...,0])
        monteCarloResults_node = numpy.stack(numpy.asarray(result)[...,1])
        monteCarloResults_glob = numpy.stack(numpy.asarray(result)[...,2])
        return monteCarloResults_elem, monteCarloResults_node, monteCarloResults_glob

    def getRandomBeamParameters_MP(self, n_simu:int):
        _ = self.process_young.getSample(n_simu)
        _ = self.process_diam.getSample(n_simu)
        _ = self.process_Rho.getSample(n_simu)
        _ = self.process_Fpos.getSample(n_simu)
        _ = self.process_Fnorm.getSample(n_simu)

    def getVonMisesStress(self,  monteCarloResults_elem=None):
        assert(self.monteCarloResults_elem is not None or monteCarloResults_elem is not None ), "first do the monte carlo experience"
        if self.monteCarloResults_elem is not None   and monteCarloResults_elem is None :
            monteCarloResults_elem = self.monteCarloResults_elem
        diameter_MC          = monteCarloResults_elem[:,2,:]
        moment_MC            = monteCarloResults_elem[:,4,:]
        shear_MC             = monteCarloResults_elem[:,3,:]
        inertia_MC, area_MC  = self.moment_inertia_PlainRoundBeam(diameter_MC)
        maxbendingStress_MC  = self.getMaximumBendingStress(moment_MC, inertia_MC, diameter_MC)
        shearStress_MC       = (4/3)*numpy.divide(shear_MC, area_MC)  
        vonMisesCriteria     = numpy.sqrt(numpy.square(maxbendingStress_MC) + numpy.multiply(numpy.square(shearStress_MC), 3))
        return vonMisesCriteria

#####################################################################################################
#####################################################################################################
#####################################################################################################

    def getResultsAsDictionary(self, youngModu_KL, diameter_KL ):
        monteCarloResults_elem = self.monteCarloResults_elem
        monteCarloResults_node = self.monteCarloResults_node
        monteCarloResults_glob = self.monteCarloResults_glob
        vonMisesStress_MC      = self.getVonMisesStress()
        ElemRange              = monteCarloResults_elem[:,0,:]
        YoungModulus           = monteCarloResults_elem[:,1,:]               
        Diameter               = monteCarloResults_elem[:,2,:]       
        Shear                  = monteCarloResults_elem[:,3,:]   
        Moment                 = monteCarloResults_elem[:,4,:] 
        PointRange             = monteCarloResults_node[:,0,:] 
        Deflection             = monteCarloResults_node[:,1,:]       
        VonMisesStress         = vonMisesStress_MC           
        AppliedForce           = monteCarloResults_glob[:,0]           
        Density                = monteCarloResults_glob[:,1]
        ForcePosition          = monteCarloResults_glob[:,2]           
        finalDictionnary = {'ElemRange'      :ElemRange,
                            'YoungModulus'   :YoungModulus,
                            'youngModulus_KL':youngModu_KL,
                            'Diameter'       :Diameter,
                            'diameter_KL'    :diameter_KL,
                            'Shear'          :Shear,
                            'Moment'         :Moment,
                            'PointRange'     :PointRange,
                            'Deflection'     :Deflection,
                            'VonMisesStress' :VonMisesStress,
                            'AppliedForce'   :AppliedForce,
                            'Density'        :Density,
                            'ForcePosition'  :ForcePosition,
                             }
        return finalDictionnary


    def saveMonteCarloResults(self, saving_path = './monteCarloExp1', shutdown=False ):
        monteCarloResults_elem = self.monteCarloResults_elem
        monteCarloResults_node = self.monteCarloResults_node
        monteCarloResults_glob = self.monteCarloResults_glob
        vonMisesStress_MC      = self.getVonMisesStress()
        ElemRange              = monteCarloResults_elem[:,0,:]
        YoungModulus           = monteCarloResults_elem[:,1,:]               
        Diameter               = monteCarloResults_elem[:,2,:]       
        Shear                  = monteCarloResults_elem[:,3,:]   
        Moment                 = monteCarloResults_elem[:,4,:] 
        PointRange             = monteCarloResults_node[:,0,:] 
        Deflection             = monteCarloResults_node[:,1,:]       
        VonMisesStress         = vonMisesStress_MC           
        AppliedForce           = monteCarloResults_glob[:,0]           
        Density                = monteCarloResults_glob[:,1]
        ForcePosition          = monteCarloResults_glob[:,2]           

        finalDictionnary = {'ElemRange'      :ElemRange,
                            'YoungModulus'   :YoungModulus,
                            'Diameter'       :Diameter,
                            'Shear'          :Shear,
                            'Moment'         :Moment,
                            'PointRange'     :PointRange,
                            'Deflection'     :Deflection,
                            'VonMisesStress' :VonMisesStress,
                            'AppliedForce'   :AppliedForce,
                            'Density'        :Density,
                            'ForcePosition'  :ForcePosition,
                             }
        file = open("./monteCarloExperimentRandomBeam2.pkl",'wb')
        pickle.dump(finalDictionnary, file)
        file.close()

        if shutdown :
            os.system('shutdown now')

    def plot_monte_carlo_res_mult(self, monteCarloResults_elem=None, monteCarloResults_node=None, monteCarloResults_glob=None, t_plt = 0.1):
        assert((self.monteCarloResults_elem is not None and self.monteCarloResults_node is not None and self.monteCarloResults_glob is not None) or \
            (monteCarloResults_elem is not None and monteCarloResults_node is not None)), "first do the monte carlo experience"
        if (self.monteCarloResults_elem is not None and self.monteCarloResults_node is not None and self.monteCarloResults_glob is not None) and \
            (monteCarloResults_elem is None and monteCarloResults_node is None and monteCarloResults_glob is None):
            monteCarloResults_elem = self.monteCarloResults_elem
            monteCarloResults_node = self.monteCarloResults_node
            monteCarloResults_glob = self.monteCarloResults_glob
            vonMisesStress_MC      = self.getVonMisesStress()
        X_elem0 = monteCarloResults_elem[0,0,:]
        X_node0 = monteCarloResults_node[0,0,:]
        images = []
        pyplot.ion()
        fig = pyplot.figure(figsize=(20,10))

        # graphs on grid
        graph1 = pyplot.subplot2grid((6,7),(0,0),colspan = 3 ,rowspan = 2 ,fig = fig) 
        graph2 = pyplot.subplot2grid((6,7),(2,0),colspan = 3 ,rowspan = 2 ,fig = fig)
        graph3 = pyplot.subplot2grid((6,7),(0,3),colspan = 3 ,rowspan = 2 ,fig = fig)
        graph4 = pyplot.subplot2grid((6,7),(2,3),colspan = 3 ,rowspan = 2 ,fig = fig)
        graph5 = pyplot.subplot2grid((6,7),(4,3),colspan = 3 ,rowspan = 2 ,fig = fig)
        graph6 = pyplot.subplot2grid((6,7),(4,0),colspan = 3 ,rowspan = 2 ,fig = fig)
        graph7 = pyplot.subplot2grid((6,7),(0,6),colspan = 1 ,rowspan = 3 ,fig = fig)
        graph8 = pyplot.subplot2grid((6,7),(3,6),colspan = 1 ,rowspan = 3 ,fig = fig)

        # define titles
        graph1.set_title('Young Modulus (MPa)'    , fontsize = 10)
        graph2.set_title('Diameter (mm)'          , fontsize = 10)
        graph3.set_title('Shear (N)'              , fontsize = 10)
        graph4.set_title('Moment (N.mm)'          , fontsize = 10)
        graph5.set_title('Deflection (mm)'        , fontsize = 10)
        graph6.set_title('Von Mises Stress (MPa)' , fontsize = 10)
        graph7.set_title('Applied force (N)'      , fontsize = 10)
        graph8.set_title('Density (kg/m³)'        , fontsize = 10)

        color_dict = {1:'r-', 2:'g--', 3:'b-.', 4:'m:', 5:'c-'}
        color_dict2= {1:'r', 2:'g', 3:'b', 4:'m', 5:'c'}
        ##define graphs
        lines1 = [graph1.plot(X_elem0, monteCarloResults_elem[i,1,:], color_dict[i+1])[0] for i in range(5)]
        lines2 = [graph2.plot(X_elem0, monteCarloResults_elem[i,2,:], color_dict[i+1])[0] for i in range(5)]
        lines3 = [graph3.plot(X_elem0, monteCarloResults_elem[i,3,:], color_dict[i+1])[0] for i in range(5)]
        lines4 = [graph4.plot(X_elem0, monteCarloResults_elem[i,4,:], color_dict[i+1])[0] for i in range(5)]
        lines5 = [graph5.plot(X_node0, monteCarloResults_node[i,1,:], color_dict[i+1])[0] for i in range(5)]
        lines6 = [graph6.plot(X_elem0, vonMisesStress_MC[i,:]       , color_dict[i+1])[0] for i in range(5)]
        barsh7 = graph7.barh(y = [0,1,2,3,4], width = monteCarloResults_glob[0:5,0], height = 0.8, align = 'edge', color = [color_dict2[i+1] for i in range(5)])  
        barsh8 = graph8.barh(y = [0,1,2,3,4], width = monteCarloResults_glob[0:5,1], height = 0.8, align = 'edge', color = [color_dict2[i+1] for i in range(5)])  


        axvlines1 = [graph1.axvline(x = monteCarloResults_glob[i,2], c = color_dict2[i+1]) for i in range(5)]
        axvlines2 = [graph2.axvline(x = monteCarloResults_glob[i,2], c = color_dict2[i+1]) for i in range(5)]
        axvlines3 = [graph3.axvline(x = monteCarloResults_glob[i,2], c = color_dict2[i+1]) for i in range(5)]
        axvlines4 = [graph4.axvline(x = monteCarloResults_glob[i,2], c = color_dict2[i+1]) for i in range(5)]
        axvlines5 = [graph5.axvline(x = monteCarloResults_glob[i,2], c = color_dict2[i+1]) for i in range(5)]
        axvlines6 = [graph6.axvline(x = monteCarloResults_glob[i,2], c = color_dict2[i+1]) for i in range(5)]
        fig.subplots_adjust(hspace=0.25,wspace=0.25)
        pyplot.tight_layout()
        fig.canvas.draw()
        images.append(fig2img(fig))
        pyplot.pause(t_plt)
        fig.canvas.flush_events()

        for i in range(5, monteCarloResults_elem.shape[0], 5):
            [lines1[k].set_xdata(monteCarloResults_elem[i+k,0,:])  for k in range(5)]
            [lines2[k].set_xdata(monteCarloResults_elem[i+k,0,:])  for k in range(5)]
            [lines3[k].set_xdata(monteCarloResults_elem[i+k,0,:])  for k in range(5)]
            [lines4[k].set_xdata(monteCarloResults_elem[i+k,0,:])  for k in range(5)]
            [lines5[k].set_xdata(monteCarloResults_node[i+k,0,:])  for k in range(5)]
            [lines6[k].set_xdata(monteCarloResults_elem[i+k,0,:])  for k in range(5)]

            [lines1[k].set_ydata(monteCarloResults_elem[i+k,1,:])  for k in range(5)]
            [lines2[k].set_ydata(monteCarloResults_elem[i+k,2,:])  for k in range(5)]
            [lines3[k].set_ydata(monteCarloResults_elem[i+k,3,:])  for k in range(5)]
            [lines4[k].set_ydata(monteCarloResults_elem[i+k,4,:])  for k in range(5)]
            [lines5[k].set_ydata(monteCarloResults_node[i+k,1,:])  for k in range(5)]
            [lines6[k].set_ydata(vonMisesStress_MC[i+k,:])         for k in range(5)]

            [axvlines1[k].set_xdata(monteCarloResults_glob[i+k,2]) for k in range(5)]
            [axvlines2[k].set_xdata(monteCarloResults_glob[i+k,2]) for k in range(5)]
            [axvlines3[k].set_xdata(monteCarloResults_glob[i+k,2]) for k in range(5)]
            [axvlines4[k].set_xdata(monteCarloResults_glob[i+k,2]) for k in range(5)]
            [axvlines5[k].set_xdata(monteCarloResults_glob[i+k,2]) for k in range(5)]
            [axvlines6[k].set_xdata(monteCarloResults_glob[i+k,2]) for k in range(5)]

            for rect, h in zip(barsh7, monteCarloResults_glob[i:i+5,0]):
                rect.set_width(h)
            for rect, h in zip(barsh8, monteCarloResults_glob[i:i+5,1]):
                rect.set_width(h)

            graph1.relim()
            graph2.relim()
            graph3.relim()
            graph4.relim()
            graph5.relim()
            graph6.relim()
            graph7.relim()
            graph8.relim()
            graph1.autoscale_view()
            graph2.autoscale_view()
            graph3.autoscale_view()
            graph4.autoscale_view()
            graph5.autoscale_view()
            graph6.autoscale_view()
            graph7.autoscale_view()
            graph8.autoscale_view()
            fig.canvas.draw()
            images.append(fig2img(fig))
            pyplot.pause(t_plt)
            fig.canvas.flush_events()

        return images

########################################################################################################################################################################################
########################################################################################################################################################################################
########################################################################################################################################################################################

def fig2data (fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h       = fig.canvas.get_width_height()
    buf       = numpy.fromstring( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
    
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll( buf, 3, axis = 2 )
    return buf

def fig2img (fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf     = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tobytes( ) )


class OpenturnsFunctionWrapperRandomBeam(openturns.OpenTURNSPythonFunction):
    def __init__(self, n_outputs=1):
        self.RandomBeamObject = RandomBeam_anastruct()
        self.inputDim    = None
        inputVariables   = self.prepareInputVarNames()
        self.n_outputs   = n_outputs 
        self.outputDim   = int(self.n_outputs*(self.RandomBeamObject.mesh.getSimplicesNumber() + 1) )#as we add a node in the code
        super(OpenturnsFunctionWrapperRandomBeam, self).__init__(self.inputDim, self.outputDim)
        self.setInputDescription(inputVariables)
        self.setOutputDescription(self.prepareOutputVarNames('VM'))
        self.composedDistribution = self.prepareComposedDistribution()

    def prepareComposedDistribution(self):
        # we have two stochastic processes 
        # we will later have this function to inspect the input function
        process_young = self.RandomBeamObject.process_young    # NdGaussianProcessConstructor class
        process_diam  = self.RandomBeamObject.process_diam     # NdGaussianProcessConstructor class
        process_Rho   = self.RandomBeamObject.process_Rho      # NormalDistribution class
        process_Fpos  = self.RandomBeamObject.process_Fpos     # NormalDistribution class
        process_Fnorm = self.RandomBeamObject.process_Fnorm    # NormalDistribution class
        
        tempList               = [process_young, process_diam, process_Rho, process_Fpos, process_Fnorm]
        processYoungAsRandVect = process_young.decompositionAsRandomVector
        processDiamAsRandVect  = process_diam.decompositionAsRandomVector
        
        distributionList       = [*processYoungAsRandVect.getRandVectorAsOtNormalsList(), *processDiamAsRandVect.getRandVectorAsOtNormalsList(), process_Rho, process_Fpos, process_Fnorm]
        composedDistribution   = openturns.ComposedDistribution(distributionList)
        return composedDistribution

    def prepareInputVarNames(self):
        # we have two stochastic processes 
        # we will later have this function to inspect the input function
        process_young = self.RandomBeamObject.process_young    # NdGaussianProcessConstructor class
        process_diam  = self.RandomBeamObject.process_diam     # NdGaussianProcessConstructor class
        process_Rho   = self.RandomBeamObject.process_Rho      # NormalDistribution class
        process_Fpos  = self.RandomBeamObject.process_Fpos     # NormalDistribution class
        process_Fnorm = self.RandomBeamObject.process_Fnorm    # NormalDistribution class

        processYoungAsRandVect = process_young.decompositionAsRandomVector
        processDiamAsRandVect  = process_diam.decompositionAsRandomVector

        inputVariables = processYoungAsRandVect.getDescription()
        inputVariables.extend(processDiamAsRandVect.getDescription())
        inputVariables.append(process_Rho.getName())
        inputVariables.append(process_Fpos.getName())
        inputVariables.append(process_Fnorm.getName())
        self.inputDim  = len(inputVariables)
        return inputVariables

    def prepareOutputVarNames(self, *varNames):
        assert len(varNames)==self.n_outputs,""
        outputVariables = list()
        for varName in varNames:
            for k in range(int(self.outputDim/self.n_outputs)):
                outputVariables.append(varName+'_'+str(k))
        return outputVariables

    def getInputDimension(self):
        assert self.composedDistribution is not None ,"use methode self.prepareComposedDistribution()"
        return self.composedDistribution.getDimension()

    def _exec(self, X):
        return self.RandomBeamObject._execute(X)

    def _exec_sample(self, X):
        return self.RandomBeamObject._execute_sample(X)



class sampleAndSoloFunctionWrapper(object):
    def __init__(self, process_E, process_D, RV_Rho, RV_FPos, RV_FNorm):
        self.RandomBeamObject = RandomBeam_anastruct(process_E,
                                                     process_D,
                                                     RV_Rho,
                                                     RV_FPos,
                                                     RV_FNorm)
        self.results = None

    
    def randomBeamFunctionSample(self,
                                 random_young_modulus, 
                                 random_diameter, 
                                 random_density, 
                                 random_forcePos, 
                                 random_forceNorm):
        result         = self.RandomBeamObject.multiprocessBatchField(random_young_modulus, random_diameter, random_density, random_forcePos, random_forceNorm )
        monteCarloResults_elem, monteCarloResults_node, monteCarloResults_glob = result
        deflection     = monteCarloResults_node[:,1,:]
        print('shape deflection: ', deflection.shape, ' should be [N,10X] something')
        vonMisesStress = self.RandomBeamObject.getVonMisesStress(monteCarloResults_elem)
        maxDeflection  = numpy.amax(numpy.abs(deflection), 1)
        print('deflection std deviation ',numpy.std(maxDeflection))
        self.results   = result
        return vonMisesStress, maxDeflection

    
    def randomBeamFunctionSolo(self,
                               random_young_modulus, 
                               random_diameter, 
                               random_density, 
                               random_forcePos, 
                               random_forceNorm):
        Result = self.RandomBeamObject.experienceBeam(random_young_modulus,
                                                      random_diameter,
                                                      random_density,
                                                      random_forcePos,
                                                      random_forceNorm)
        deflection, shear, moment = self.RandomBeamObject.postprecess_beam(returnStuff = True)
        inertia, area             = self.RandomBeamObject.moment_inertia_PlainRoundBeam(numpy.squeeze(numpy.asarray(field_diam)))
        maxbendingStress          = self.RandomBeamObject.getMaximumBendingStress(moment, inertia, numpy.squeeze(numpy.asarray(field_diam)))
        shearStress               = (4/3)*numpy.divide(shear, area)  
        vonMisesCriteria          = numpy.sqrt(numpy.square(maxbendingStress) + numpy.multiply(numpy.square(shearStress), 3))
        maxDeflection             = numpy.abs(np.min(deflection))
        return vonMisesCriteria, maxDeflection     


