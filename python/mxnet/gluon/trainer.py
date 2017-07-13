# coding: utf-8
# pylint: disable=
"""Parameter optimizer."""

from .. import optimizer as opt
from ..model import _create_kvstore

class Trainer(object):
    """Applies an Optimizer on a set of Parameters. Trainer should
    be used together with autograd.

    Parameters
    ----------
    params : ParameterDict
        The set of parameters to optimize.
    optimizer : str or Optimizer
        The optimizer to use.
    optimizer_params : dict
        key-word arguments to be passed to Optimizer.create_optimizer. For example,
        {'learning_rate': 0.1}
    kvstore : str or KVStore
        kvstore type for multi-gpu and distributed training.
    """
    def __init__(self, params, optimizer, optimizer_params, kvstore='device'):
        self._params = [param for param in params.values() if param.grad_req != 'null']
        self._scale = optimizer_params.get('rescale_grad', 1.0)

        self._contexts = self._check_contexts()
        self._init_optimizer(optimizer, optimizer_params)
        self._kv_initialized = False
        self._kvstore = kvstore

    def _check_contexts(self):
        contexts = None
        for param in self._params:
            ctx = param.list_ctx()
            assert contexts is None or contexts == ctx, \
                "All Parameters must be initialized on the same set of contexts, " \
                "but Parameter %s is initialized on %s while previous Parameters " \
                "are initialized on %s."%(param.name, str(ctx), str(contexts))
            contexts = ctx
        return contexts

    def _init_optimizer(self, optimizer, optimizer_params):
        self._optimizer = opt.create(optimizer, **optimizer_params)

        lr_mult = {}
        wd_mult = {}
        for i, param in enumerate(self._params):
            lr_mult[i] = param.lr_mult
            wd_mult[i] = param.wd_mult
        self._optimizer.set_lr_mult(lr_mult)
        self._optimizer.set_wd_mult(wd_mult)

        self._updaters = [opt.get_updater(self._optimizer) \
                            for _ in self._contexts]

    def _init_kvstore(self):
        arg_arrays = {param.name: param.data(self._contexts[0]) for param in self._params}
        kvstore, update_on_kvstore = _create_kvstore(self._kvstore, len(self._contexts), arg_arrays)
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        if kvstore:
            assert 'dist' not in self._kvstore.type, "distributed training not supported yet"
            for i, param in enumerate(self._params):
                param_arrays = param.list_data()
                kvstore.init(i, param_arrays[0])
                kvstore.pull(i, param_arrays, priority=-i)
            if update_on_kvstore:
                kvstore.set_optimizer(self._optimizer)
        self._kv_initialized = True

    def step(self, batch_size, ignore_stale_grad=False):
        """Make one step of parameter update. Should be called after
        autograd.compute_gradient and outside of record() scope.

        Parameters
        ----------
        batch_size : int
            Batch size of data processed. Gradient will be normalized by 1/batch_size.
            Set this to 1 if you normalized loss manually with `loss = mean(loss)`.
        ignore_stale_grad : bool, optional, default=False
            If true, ignores Parameters with stale gradient (gradient that has not
            been updated by `backward` after last step) and skip update.
        """
        if not self._kv_initialized:
            self._init_kvstore()

        self._optimizer.rescale_grad = self._scale / batch_size

        for i, param in enumerate(self._params):
            assert param.list_ctx() == self._contexts, \
                "Parameter %s's contexts changed after Optim initialization: " \
                "was %s, now %s"%(param.name, self._contexts, param.list_ctx())
            if not ignore_stale_grad:
                for data in param.list_data():
                    if not data._fresh_grad:
                        raise UserWarning(
                            "Gradient of Parameter `%s` on context %s has not been updated "
                            "by backward since last `step`. This could mean a bug in your "
                            "model that maked it only use a subset of the Parameters (Blocks) "
                            "for this iteration. If you are intentionally only using a subset, "
                            "call step with ignore_stale_grad=True to suppress this "
                            "warning and skip updating of Parameters with stale gradient" \
                            %(param.name, str(data.context)))
            if self._kvstore:
                self._kvstore.push(i, param.list_grad(), priority=-i)
                if self._update_on_kvstore:
                    self._kvstore.pull(i, param.list_data(), priority=-i)
                    continue
                else:
                    self._kvstore.pull(i, param.list_grad(), priority=-i)
            for upd, arr, grad in zip(self._updaters, param.list_data(), param.list_grad()):
                if arr._fresh_grad:
                    upd(i, grad, arr)
                    arr._fresh_grad = False
