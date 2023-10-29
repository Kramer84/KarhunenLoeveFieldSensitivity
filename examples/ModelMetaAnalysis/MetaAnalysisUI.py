import gc
import os, sys, re
import time
import numpy as np
import pandas as pd

import warnings
import openturns as ot

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import matplotlib as mpl

mpl.use("Qt4Agg")

from matplotlib.figure import Figure

from traits.api import (
    HasTraits,
    Bool,
    Event,
    File,
    Int,
    Str,
    String,
    Directory,
    Function,
    Color,
    Enum,
    List,
    Button,
    Range,
    Instance,
    Float,
    Trait,
    Any,
    CFloat,
    Property,
    Either,
    on_trait_change,
)

from traitsui.api import (
    Handler,
    View,
    Item,
    OKCancelButtons,
    OKButton,
    CancelButton,
    Spring,
    InstanceEditor,
    Group,
    ListStrEditor,
    CheckListEditor,
    HSplit,
    FileEditor,
    VSplit,
    Action,
    HGroup,
    TextEditor,
    ImageEnumEditor,
    UIInfo,
    Label,
    VGroup,
    ListEditor,
    TableEditor,
    ObjectColumn,
    WindowColor,
    message,
    auto_close_message,
    message,
    BooleanEditor,
    EnumEditor,
)

from pyface.qt import QtGui, QtCore

from traitsui.qt4.editor import Editor
from traitsui.qt4.basic_editor_factory import BasicEditorFactory

from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from traits.api import (
    HasTraits,
    Bool,
    Event,
    File,
    Int,
    Str,
    String,
    Directory,
    Function,
    Color,
    Enum,
    List,
    Button,
    Range,
    Instance,
    Float,
    Trait,
    Any,
    CFloat,
    Property,
    Either,
    on_trait_change,
)

from traitsui.api import (
    Handler,
    View,
    Item,
    OKCancelButtons,
    OKButton,
    CancelButton,
    Spring,
    InstanceEditor,
    Group,
    ListStrEditor,
    CheckListEditor,
    HSplit,
    FileEditor,
    VSplit,
    Action,
    HGroup,
    TextEditor,
    ImageEnumEditor,
    UIInfo,
    Label,
    VGroup,
    ListEditor,
    TableEditor,
    ObjectColumn,
    WindowColor,
    message,
    auto_close_message,
    message,
    BooleanEditor,
    EnumEditor,
)


class TextDisplay(HasTraits):
    string = String()
    view = View(Item("string", show_label=False, springy=True, style="custom"))


class _MPLFigureEditor(Editor):
    scrollable = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """Create the MPL canvas."""
        # matplotlib commands to create a canvas
        frame = QtGui.QWidget()
        mpl_canvas = FigureCanvas(self.value)
        mpl_canvas.setParent(frame)
        mpl_toolbar = NavigationToolbar2QT(mpl_canvas, frame)

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(mpl_canvas)
        vbox.addWidget(mpl_toolbar)
        frame.setLayout(vbox)

        return frame


class _MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor


class interfaceGUIHandler(Handler):
    ## Look in the traitsui.api handler script for infos
    def init_info(self, info):
        ## Here we create the UIInfo object
        pass

    def init(self, info):
        """This method gets called after the controls have all been
        created but before they are displayed.
        """
        info.object.mpl_setup()
        return True

    def close(self, info, is_ok=True):
        ## This happens when you click on the top right cross
        confirmation = message(
            message="Are you sure to close the window?",
            title="Warning!",
            buttons=["OK", "Cancel"],
        )
        if confirmation is True:
            return True
        else:
            return False

    def closed(self, info, is_ok):
        ## This is to do clean-up after having destroyed (disposed) the window
        ## THIS IS TO BE DONE !!!!!
        pass

    def object_destroy_changed(self, info):
        if info.initialized:
            info.bind_context()  ## Does nothing (to me)
            info.ui.dispose()


def loadMetaAnalysisResults(metaAnalyisPath="./meta_analysis_results"):
    """Function to load the data of our simulation"""
    analysis_dirs = os.listdir(metaAnalyisPath)

    lhs_params = ["young", "scaleYoung", "diam", "scaleDiam", "forcePos", "forceNorm"]
    size_lhs = len(analysis_dirs)
    lhs_doe = np.zeros((size_lhs, 6))
    for i in range(size_lhs):
        lhs_doe[i, ...] = re.findall(r"[-+]?\d*\.\d+|\d+", analysis_dirs[i])

    # size_lhs, n_nu, n_thresh, sobol_E, sobol_D, sobol_FP, sobol_FN
    # for the model we have only one evaluation
    meta_analysis_array_model = np.zeros((size_lhs, 3, 3, 4, 3))
    # for the metamodel we have an envaluation with 3 different LHS sizes
    meta_analysis_array_mm1000 = np.zeros((size_lhs, 3, 3, 3, 4, 3))
    meta_analysis_array_mm50000 = np.zeros((size_lhs, 3, 3, 3, 4, 3))
    # size_lhs, n_nu, n_thresh, kl_dim
    kl_dimension_array = np.zeros((size_lhs, 3, 3))
    nu_array = np.zeros((size_lhs, 3, 3))
    thresh_array = np.zeros((size_lhs, 3, 3))
    for k_lhs in range(size_lhs):
        thresh_paths = os.listdir(os.path.join(metaAnalyisPath, analysis_dirs[k_lhs]))
        for k_thresh in range(3):
            thresh_path = thresh_paths[k_thresh]
            thresh_val = re.findall("-?\d+.?\d*(?:[Ee]-\d+)?", thresh_path)[0]
            nu_paths = os.listdir(
                os.path.join(
                    metaAnalyisPath, analysis_dirs[k_lhs], thresh_paths[k_thresh]
                )
            )
            for k_nu in range(3):
                nu_path = nu_paths[k_nu]
                nu_val = re.findall(r"[-+]?\d*\.\d+|\d+", nu_path)[0]
                csv_path = os.path.join(
                    metaAnalyisPath, analysis_dirs[k_lhs], thresh_path, nu_path
                )
                sample = ot.Sample_ImportFromCSVFile(csv_path)
                kl_dimension_array[k_lhs, k_nu, k_thresh] = sample[0, 3]
                nu_array[k_lhs, k_nu, k_thresh] = nu_val
                thresh_array[k_lhs, k_nu, k_thresh] = thresh_val
                meta_analysis_array_model[k_lhs, k_nu, k_thresh, ...] = np.reshape(
                    np.array(sample[0, 4:]), (4, 3)
                )
                # Here we iterate over the 3 LHS sizes (25,50,100)
                for i_lhs in range(3):
                    meta_analysis_array_mm1000[
                        k_lhs, k_nu, k_thresh, i_lhs, ...
                    ] = np.reshape(np.array(sample[4 + i_lhs, 4:]), (4, 3))
                    meta_analysis_array_mm50000[
                        k_lhs, k_nu, k_thresh, i_lhs, ...
                    ] = np.reshape(np.array(sample[1 + i_lhs, 4:]), (4, 3))
    return (
        lhs_params,
        lhs_doe,
        meta_analysis_array_model,
        meta_analysis_array_mm1000,
        meta_analysis_array_mm50000,
        kl_dimension_array,
        nu_array,
        thresh_array,
    )


def set_indices_figure(
    fig,
    sobol_model,
    err_model,
    sobol_metaLHS25,
    err_metaLHS25,
    sobol_metaLHS50,
    err_metaLHS50,
    sobol_metaLHS100,
    err_metaLHS100,
):
    xlabels = [
        "Sobol Young",
        "Sobol Diameter",
        "Sobol Force Position",
        "Sobol Force Norm",
    ]
    x_val = np.array([0, 1, 2, 3])
    ax = fig.add_subplot(111)
    ax = fig.axes[0]
    offset = 0.1
    sobol_model = ax.errorbar(
        x=x_val,
        y=sobol_model,
        yerr=[
            np.absolute(sobol_model - err_model[0, ...]),
            np.absolute(sobol_model - err_model[1, ...]),
        ],
        fmt="s",
        color="navy",
        ecolor="navy",
        label="Model",
    )

    sobol_meta_LHS25 = ax.errorbar(
        x=x_val + offset,
        y=sobol_metaLHS25,
        yerr=[
            np.absolute(sobol_metaLHS25 - err_metaLHS25[0, ...]),
            np.absolute(sobol_metaLHS25 - err_metaLHS25[1, ...]),
        ],
        fmt="s",
        color="yellow",
        ecolor="yellow",
        label="LHS25",
    )
    sobol_meta_LHS50 = ax.errorbar(
        x=x_val + 2 * offset,
        y=sobol_metaLHS50,
        yerr=[
            np.absolute(sobol_metaLHS50 - err_metaLHS50[0, ...]),
            np.absolute(sobol_metaLHS50 - err_metaLHS50[1, ...]),
        ],
        fmt="s",
        color="darkorange",
        ecolor="darkorange",
        label="LHS50",
    )
    sobol_meta_LHS100 = ax.errorbar(
        x=x_val + 3 * offset,
        y=sobol_metaLHS100,
        yerr=[
            np.absolute(sobol_metaLHS100 - err_metaLHS100[0, ...]),
            np.absolute(sobol_metaLHS100 - err_metaLHS100[1, ...]),
        ],
        fmt="s",
        color="red",
        ecolor="red",
        label="LHS100",
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(x_val + 0.2)
    ax.set_xticklabels(xlabels)
    ax.legend(loc="upper right")
    return fig


class metaAnalysisVisualizer(HasTraits):
    """This is the class that handles the GUI, there will be some buttons and the matplotlib figure"""

    index_lhs = Int(-1)
    index_nu = Int(0)
    index_thresh = Int(0)
    index_eval_size = Int(0)

    threshold_value = Float(0)
    nu_value = Float(0)
    size_sobol_expriment_metamodel = Int(1000)

    young_modulus_params = Str("")
    diameter_params = Str("")
    force_norm_params = Str("")
    force_pos_params = Str("")
    kl_size_params = Int(0)

    figure = Instance(Figure, ())
    next_realization = Button(label="Select next realization")
    previous_realization = Button(label="Select previous realization")
    change_threshold = Button(label="change threshold")
    change_nu = Button(label="change NU")
    change_sobol_size_LHS = Button(label="change sobol experiment size metamodel")

    view = View(
        HSplit(
            VSplit(
                VGroup(
                    Item("next_realization", show_label=False, height=0.1),
                    Item("previous_realization", show_label=False, height=0.1),
                    Item("change_threshold", show_label=False, height=0.1),
                    Item("change_nu", show_label=False, height=0.1),
                    Item("change_sobol_size_LHS", show_label=False, height=0.1),
                ),
                VGroup(
                    Item(
                        "young_modulus_params",
                        show_label=True,
                        label="E (VAR / SCALE) :",
                        style="readonly",
                    ),
                    Item(
                        "diameter_params",
                        show_label=True,
                        label="D (VAR / SCALE) :",
                        style="readonly",
                    ),
                    Item(
                        "force_norm_params",
                        show_label=True,
                        label="FN (VAR) :",
                        style="readonly",
                    ),
                    Item(
                        "force_pos_params",
                        show_label=True,
                        label="FP (VAR) :",
                        style="readonly",
                    ),
                    Item(
                        "kl_size_params",
                        show_label=True,
                        label="Size KL :",
                        style="readonly",
                    ),
                ),
                VGroup(
                    Item("index_lhs", show_label=True, style="readonly"),
                    Item(
                        "threshold_value",
                        show_label=True,
                        label="Threshold",
                        style="readonly",
                    ),
                    Item("nu_value", show_label=True, label="Nu", style="readonly"),
                    Item(
                        "size_sobol_expriment_metamodel",
                        show_label=True,
                        label="Experiment size MM",
                        style="readonly",
                    ),
                ),
            ),
            Item("figure", editor=_MPLFigureEditor(), show_label=False, width=0.75),
        ),
        handler=interfaceGUIHandler(),
        resizable=True,
        scrollable=False,
        height=1,
        width=1,
        title="Blood Analysis Interface",
        icon="Interface",
    )

    def __init__(self):
        self.xlabels = ["SE", "SD", "SFN", "SFP"]
        output = loadMetaAnalysisResults()
        self.lhs_params = output[0]
        self.lhs_doe = output[1]
        self.meta_analysis_array_model = output[2]
        self.meta_analysis_array_mm1000 = output[3]
        self.meta_analysis_array_mm50000 = output[4]
        self.kl_dimension_array = output[5]
        self.nu_array = output[6]
        self.thresh_array = output[7]

    def _figure_default(self):
        """Initialises the display."""
        x_val = np.array([0, 1, 2, 3])
        figure = Figure()
        figure = set_indices_figure(
            figure,
            np.zeros((4,)) + 0.25,
            np.ones((2, 4)) * 0.1,
            np.zeros((4,)) + 0.25,
            np.ones((2, 4)) * 0.1,
            np.zeros((4,)) + 0.25,
            np.ones((2, 4)) * 0.1,
            np.zeros((4,)) + 0.25,
            np.ones((2, 4)) * 0.1,
        )
        print("Figure initialized")

        # Set matplotlib canvas colour to be white
        rect = figure.patch
        rect.set_facecolor([0.6, 0.6, 0.6])
        return figure

    def getDataFromIndex(self):
        if (
            self.index_lhs >= 0
            and self.index_nu >= 0
            and self.index_thresh >= 0
            and self.index_eval_size == 0
        ):
            sobol_model = self.meta_analysis_array_model[
                self.index_lhs, self.index_nu, self.index_thresh, :, 0
            ]
            err_model = self.meta_analysis_array_model[
                self.index_lhs, self.index_nu, self.index_thresh, :, 1:
            ]
            err_model = np.array([err_model[:, 0], err_model[:, 1]])

            sobol_metaLHS25 = self.meta_analysis_array_mm1000[
                self.index_lhs, self.index_nu, self.index_thresh, 0, :, 0
            ]
            err_metaLHS25 = self.meta_analysis_array_mm1000[
                self.index_lhs, self.index_nu, self.index_thresh, 0, :, 1:
            ]
            err_metaLHS25 = np.array([err_metaLHS25[:, 0], err_metaLHS25[:, 1]])

            sobol_metaLHS50 = self.meta_analysis_array_mm1000[
                self.index_lhs, self.index_nu, self.index_thresh, 1, :, 0
            ]
            err_metaLHS50 = self.meta_analysis_array_mm1000[
                self.index_lhs, self.index_nu, self.index_thresh, 1, :, 1:
            ]
            err_metaLHS50 = np.array([err_metaLHS50[:, 0], err_metaLHS50[:, 1]])

            sobol_metaLHS100 = self.meta_analysis_array_mm1000[
                self.index_lhs, self.index_nu, self.index_thresh, 2, :, 0
            ]
            err_metaLHS100 = self.meta_analysis_array_mm1000[
                self.index_lhs, self.index_nu, self.index_thresh, 2, :, 1:
            ]
            err_metaLHS100 = np.array([err_metaLHS100[:, 0], err_metaLHS100[:, 1]])

            return (
                sobol_model,
                err_model,
                sobol_metaLHS25,
                err_metaLHS25,
                sobol_metaLHS50,
                err_metaLHS50,
                sobol_metaLHS100,
                err_metaLHS100,
            )

        elif (
            self.index_lhs >= 0
            and self.index_nu >= 0
            and self.index_thresh >= 0
            and self.index_eval_size == 1
        ):
            sobol_model = self.meta_analysis_array_model[
                self.index_lhs, self.index_nu, self.index_thresh, :, 0
            ]
            err_model = self.meta_analysis_array_model[
                self.index_lhs, self.index_nu, self.index_thresh, :, 1:
            ]
            err_model = np.array([err_model[:, 0], err_model[:, 1]])

            sobol_metaLHS25 = self.meta_analysis_array_mm50000[
                self.index_lhs, self.index_nu, self.index_thresh, 0, :, 0
            ]
            err_metaLHS25 = self.meta_analysis_array_mm50000[
                self.index_lhs, self.index_nu, self.index_thresh, 0, :, 1:
            ]
            err_metaLHS25 = np.array([err_metaLHS25[:, 0], err_metaLHS25[:, 1]])

            sobol_metaLHS50 = self.meta_analysis_array_mm50000[
                self.index_lhs, self.index_nu, self.index_thresh, 1, :, 0
            ]
            err_metaLHS50 = self.meta_analysis_array_mm50000[
                self.index_lhs, self.index_nu, self.index_thresh, 1, :, 1:
            ]
            err_metaLHS50 = np.array([err_metaLHS50[:, 0], err_metaLHS50[:, 1]])

            sobol_metaLHS100 = self.meta_analysis_array_mm50000[
                self.index_lhs, self.index_nu, self.index_thresh, 2, :, 0
            ]
            err_metaLHS100 = self.meta_analysis_array_mm50000[
                self.index_lhs, self.index_nu, self.index_thresh, 2, :, 1:
            ]
            err_metaLHS100 = np.array([err_metaLHS100[:, 0], err_metaLHS100[:, 1]])

            return (
                sobol_model,
                err_model,
                sobol_metaLHS25,
                err_metaLHS25,
                sobol_metaLHS50,
                err_metaLHS50,
                sobol_metaLHS100,
                err_metaLHS100,
            )

        else:
            # dummy sobol / dummy error
            ds = np.zeros((4,)) + 0.25
            de = np.ones((2, 4)) * 0.1
            return ds, de, ds, de, ds, de, ds, de

    def regenerate_plot(self):
        self.young_modulus_params = (
            str(round(self.lhs_doe[self.index_lhs, 0] / 210000, 4))
            + "  /  "
            + str(round(self.lhs_doe[self.index_lhs, 1] / 1000, 4))
        )
        self.diameter_params = (
            str(round(self.lhs_doe[self.index_lhs, 2] / 10, 4))
            + "  /  "
            + str(round(self.lhs_doe[self.index_lhs, 3] / 1000, 4))
        )
        self.force_norm_params = str(round(self.lhs_doe[self.index_lhs, 5] / 100, 4))
        self.force_pos_params = str(round(self.lhs_doe[self.index_lhs, 4] / 500, 4))
        self.kl_size_params = int(
            self.kl_dimension_array[self.index_lhs, self.index_nu, self.index_thresh]
        )
        (
            sobol_model,
            err_model,
            sobol_metaLHS25,
            err_metaLHS25,
            sobol_metaLHS50,
            err_metaLHS50,
            sobol_metaLHS100,
            err_metaLHS100,
        ) = self.getDataFromIndex()
        figure = self.figure
        figure.clear()
        set_indices_figure(
            figure,
            sobol_model,
            err_model,
            sobol_metaLHS25,
            err_metaLHS25,
            sobol_metaLHS50,
            err_metaLHS50,
            sobol_metaLHS100,
            err_metaLHS100,
        )
        canvas = self.figure.canvas
        if canvas is not None:
            canvas.draw()

    def _init_index(self):
        if self.index_lhs == -1:
            self.index_lhs = 0
        self.threshold_value = self.thresh_array[
            self.index_lhs, self.index_nu, self.index_thresh
        ]
        self.nu_value = self.nu_array[self.index_lhs, self.index_nu, self.index_thresh]

    def _index_lhs_changed(self):
        self.regenerate_plot()

    def _index_thresh_changed(self):
        self.regenerate_plot()
        self.threshold_value = self.thresh_array[
            self.index_lhs, self.index_nu, self.index_thresh
        ]

    def _index_eval_size_changed(self):
        self.regenerate_plot()

    def _index_nu_changed(self):
        self.regenerate_plot()
        self.nu_value = self.nu_array[self.index_lhs, self.index_nu, self.index_thresh]

    def _next_realization_fired(self):
        self._init_index()
        self.index_lhs = (self.index_lhs + 1) % self.lhs_doe.shape[0]

    def _previous_realization_fired(self):
        self._init_index()
        self.index_lhs = (self.index_lhs - 1) % self.lhs_doe.shape[0]

    def _change_sobol_size_LHS_fired(self):
        self._init_index()
        if self.index_eval_size == 0:
            self.index_eval_size = 1
            self.size_sobol_expriment_metamodel = 50000
        else:
            self.index_eval_size = 0
            self.size_sobol_expriment_metamodel = 1000

    def _change_threshold_fired(self):
        self._init_index()
        self.index_thresh = (self.index_thresh + 1) % 3

    def _change_nu_fired(self):
        self._init_index()
        self.index_nu = (self.index_nu + 1) % 3

    def mpl_setup(self):
        pass


if __name__ == "__main__":
    view = metaAnalysisVisualizer()
    view.configure_traits()


"""
import meta_model_analysis_visualization as mmav
X = mmav.metaAnalysisVisualizer()
X.configure_traits()
"""
