import pytensor
from pytensor.graph.utils import graph_replace
from pytensor.tensor.type import vector
from tests.graph.utils import MyOp, MyVariable


def test_stack_trace():
    with pytensor.config.change_flags(traceback__limit=1):
        v = vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 1

    with pytensor.config.change_flags(traceback__limit=2):
        v = vector()
        assert len(v.tag.trace) == 1
        assert len(v.tag.trace[0]) == 2


def test_replacements():
    x = MyVariable("x")
    y = MyVariable("y")
    z = MyVariable("z")
    x2 = MyOp("xop")(x, z)
    x2.name = "x2"
    y2 = MyOp("yop")(y)
    y2.name = "y2"

    yc = graph_replace([x2], {x: y2})[0]
    assert yc.owner.inputs[0] is y2
    # the old reference is kept
    assert yc.owner.inputs[1] is z

    # the case where inputs have to be replaced in reverse topological order
    o = MyOp("xyop")(x2, y2)
    new_x = x.clone()
    new_y2 = y2.clone()

    oc = graph_replace([o], {x: new_x, y2: new_y2})[0]
    assert oc.owner.inputs[1] is new_y2
    assert oc.owner.inputs[0].owner.inputs[0] is new_x
    # the old reference is still kept
    assert oc.owner.inputs[0].owner.inputs[1] is z
