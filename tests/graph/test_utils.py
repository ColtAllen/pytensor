import pytest

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


def test_graph_replace():
    x = MyVariable("x")
    y = MyVariable("y")
    z = MyVariable("z")
    w = MyVariable("w")
    MyOp("zop")(z)
    x2 = MyOp("xop")(x, w)
    x2.name = "x2"
    y2 = MyOp("yop")(y)
    y2.name = "y2"

    yc = graph_replace([x2], {x: y2})[0]
    assert yc.owner.inputs[0] is y2
    # the old reference is kept
    assert yc.owner.inputs[1] is w

    # test replace itself
    yc = graph_replace([x2], {x2: y2})[0]
    assert yc is y2
    assert yc.owner.inputs[0] is y
    assert len(yc.owner.inputs) == 1

    # the case where inputs have to be replaced in reverse topological order
    o = MyOp("xyop")(x2, y2)
    new_x = x.clone(name="x_new")
    new_y2 = y2.clone(name="y2_new")

    oc = graph_replace([o], {x: new_x, y2: new_y2})[0]
    assert oc.owner.inputs[1] is new_y2
    assert oc.owner.inputs[0].owner.inputs[0] is new_x
    # the old reference is still kept
    assert oc.owner.inputs[0].owner.inputs[1] is w


def test_graph_replace_advanced():
    x = MyVariable("x")
    y = MyVariable("y")
    z = MyVariable("z")
    w = MyVariable("w")
    z2 = MyOp("zop")(z)
    x2 = MyOp("xop")(x, w)
    x2.name = "x2"
    y2 = MyOp("yop")(y)
    y2.name = "y2"
    o = MyOp("xyop")(x2, y2)
    new_x = x.clone(name="x_new")
    new_y2 = y2.clone(name="y2_new")
    new_y21 = MyOp("ny2op")(new_y2)
    # now yet another replacement that could only appear after new_y2: z
    # show we can do that after the prev clone
    # the case where new variable is referenced during the replacements
    new_y21 = MyOp("ny2op")(new_y2)
    # the reference new_y2: z2 is not a part of the original graph so the replacement is unsafe
    with pytest.raises(ValueError) as err:
        oc = graph_replace([o], {x: new_x, new_y2: z2, y2: new_y21})
    assert err.match("Some replacements were not used")
    oc = graph_replace([o], {x: new_x, y2: new_y21})
    oc = graph_replace(oc, {new_y2: z2})[0]
    assert oc.owner.inputs[1].owner.inputs[0] is z2
    assert oc.owner.inputs[0].owner.inputs[0] is new_x
    # the old reference is still kept
    assert oc.owner.inputs[0].owner.inputs[1] is w

    new_z = z.clone(name="z_new")
    oc = graph_replace([oc], {z: new_z})[0]
    # new reference appear
    assert oc.owner.inputs[1].owner.inputs[0] is not z2
    assert oc.owner.inputs[1].owner.inputs[0].owner.inputs[0] is new_z
    # the old reference is still kept
    assert oc.owner.inputs[0].owner.inputs[0] is new_x
    assert oc.owner.inputs[0].owner.inputs[1] is w
    # order them messed up so the correct toposort is required
    # and repeat the replacement

    fake = MyOp("fake")(x)
    oc, unused = graph_replace([o], {fake: new_x}, strict=False, return_unused=True)
    assert oc[0] is o
    assert fake in unused
