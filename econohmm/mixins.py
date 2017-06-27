__author__ = 'keithblackwell1'

import numpy
import numpy as np
from copy import deepcopy

# Todo: this really should be AccessComponentsMixin :(
# Todo: should this be an extension of a list
class ComponentizeMixin(object):

    """
    Mixin that will search for methods and fields of the components that make
    up an emission model and will run them on all the members if there_col is no
    existing definition

    all emission models must have self.components

    EM[0] = self.components[0]

    EM.mu will return [em.mu for em in EM]

    EM.method(x) will return [em.method(x) for em in EM] or None
    methods with variable inputs have to set at the individual level

    """

    def __getitem__(self, key):
        return self.components[key]

    def __getattr__(self, name):

        comps = self.components
        c1 = comps[0]
        a1 = c1.__getattribute__(name)

        #if it's a field it will output the values
        if not hasattr(a1, '__call__'):
            out = [a1]
            for cc in comps[1:]:
                out.append(cc.__getattribute__(name))
            return out

        #if method it will run the method
        else:
            ## returns a function that will run method on all the components
            def run_fun(*args, **kwargs):
                run_out = []

                for cc in comps:
                    attr = cc.__getattribute__(name)
                    run_out.append(attr(*args, **kwargs))

                if run_out[-1] is not None:
                    return run_out

            return run_fun

    def get_field_array(self, field):
        """
        This is like the lazy __getattr__ except you must choose the field or
        property, which must be a number or vector valued property
        field_array returns a numpy array

        :param field: string
        :return: numpy.ndarray
        """

        comps = self.components
        a0 = comps[0].__getattribute__(field)

        if isinstance(a0, int) or isinstance(a0, float):
            out = np.empty(self.K)

        elif isinstance(a0, numpy.ndarray):
            ll = list(a0.shape)
            ll.insert(0, self.K)
            out = np.empty(ll)

        else:
            pass

        out[0] = a0

        for i, cc in enumerate(comps[1:]):
            out[1+i] = cc.__getattribute__(field)

        return out


#Todo: This really should be ParameterHistoryMixin :(
class HistorizeMixin(object):
    """
    Mixin that allows individual paramater tracking during algorithms
    will save the output for a Gibbs Sampler or EM. All Children must set
    _hist_fields as a class variable for default values
    """
    _hist_fields = None

    def hist_init(self, N=100, fields=None, extend=False, atype=np.zeros):
        """
        creates a dictionary of tracked values
        :param N: number of iterations
        :param fields: fields to track
        :param extend: will extend the run and just add to what we've done thus far
        if extend is True and there_col is already a history then it will ignore fields and just
        :return:
        """
        assert self._hist_fields is not None
        if extend and self._hist:
            fields = self._hist.keys()
            new_run = False

        else:
            if fields is None:
                fields = self._hist_fields

            self._hist = {}
            new_run = True

        for field in fields:

            field_val = self.__getattribute__(field)
            new_array =  self.__init_array(field_val, N, atype=atype)

            if new_run:
                new_array[0] = field_val
                self._hist[field] = new_array
                self._hist['N'] = N

            else:
                old_array = self._hist[field]
                self._hist[field] = np.append(old_array, new_array, axis=0)
                self._hist['N'] += N

    @property
    def hist(self):
        if not self._hist:
            self.hist_init()
        return self._hist

    def update_fields(self, field, val, i):
        self._hist[field][i] = val

    def update_hist(self, i):
        pass

    def copy_hist(self):
        return deepcopy(self._hist)

    def __init_array(self, field_val, N, atype=np.empty):

        if isinstance(field_val, int) or isinstance(field_val, float):
            out = atype(N)

        elif isinstance(field_val, numpy.ndarray):
            ll = list(field_val.shape)
            ll.insert(0, N)
            out = atype(ll)

        else:
            raise "field not available for historizing"

        return out


class ContainerMixin(HistorizeMixin, ComponentizeMixin):
    """
    adds a component wide hist_init. This allows tracking of both individual components
    variables and container level variables
    """
    components = None

    def comp_hist_init(self, N=100, fields=None, extend=False ,atype=np.empty):
        assert self.components is not None
        for comp in self.components:
            comp.hist_init(N=N, fields=fields, extend=extend, atype=atype)

    def copy_comp_hist(self):
        return [cc.copy_hist() for cc in self.components]



# class Component(HistorizeMixin):
#
#     k = 0
#
#     _hist_fields = ['y', 'i']
#
#     def __init__(self, n=5, i=0):
#         self.x = np.random.randn(n)
#         self.y = np.random.randint(10)
#         self._n = n
#         self.n = 2
#         self.i = i
#         self.zz = None
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(i=%i)' % (self.i)
#
#     def show(self, i=0):
#         print self.x[i]
#
#     def get_x(self):
#         return self.x
#
#     def resample(self, x=3):
#         self.x = np.random.randn(self._n)
#         self.y = np.random.randint(10)
#
#     def give_i(self, i):
#         self.i = i
#         return i
#
#
# class Container(ComponentizeMixin, ContainerHistorizeMixin):
#
#     _hist_fields = ['hh']
#
#     def __init__(self, k=5, n=5):
#         self.components = [Component(i=j, n=n) for j in xrange(k)]
#         self.k = k
#         self.n = n
#         self.hh = 4
#
#     def get_components(self):
#         return self.components
#
#     def update(self):
#         self.components = [Component(i=j, n=self.n) for j in xrange(self.k)]
#
#     def get_x(self):
#         comps = self.components
#         return [1 for c in comps]
