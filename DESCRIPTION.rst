PyTensor is a Python library that allows you to define, optimize, and efficiently evaluate mathematical expressions involving multi-dimensional arrays. It is built on top of NumPy_. PyTensor features:

 * **tight integration with NumPy:** a similar interface to NumPy's. numpy.ndarrays are also used internally in PyTensor-compiled functions.
 * **efficient symbolic differentiation:** PyTensor can compute derivatives for functions of one or many inputs.
 * **speed and stability optimizations:** avoid nasty bugs when computing expressions such as log(1 + exp(x)) for large values of x.
 * **dynamic C code generation:** evaluate expressions faster.
 * **extensive unit-testing and self-verification:** includes tools for detecting and diagnosing bugs and/or potential problems.

.. _NumPy: http://numpy.scipy.org/
