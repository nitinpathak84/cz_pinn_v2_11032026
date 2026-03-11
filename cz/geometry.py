from dataclasses import dataclass

from sympy import Symbol
from physicsnemo.sym.geometry.primitives_2d import Rectangle


@dataclass
class CzGeometry:
    x: object
    y: object

    crystal: object
    melt: object
    crucible: object
    argon: object
    heater: object
    insulation: object
    outer_box: object
    argon_box: object

    r_crystal: float
    h_crystal: float

    r_melt: float
    h_melt: float

    t_crucible_wall: float
    t_crucible_bottom: float
    r_crucible_outer: float

    heater_gap: float
    heater_thickness: float
    r_heater_in: float
    r_heater_out: float
    y_heater_bot: float
    y_heater_top: float

    r_outer: float
    y_outer_bot: float
    y_outer_top: float

    y_sl: float
    y_crystal_top: float
    y_argon_top: float


def build_cz_geometry(cfg) -> CzGeometry:
    g = cfg.custom.geometry

    x = Symbol("x")
    y = Symbol("y")

    r_crystal = float(g.crystal_radius)
    h_crystal = float(g.crystal_height)

    r_melt = float(g.melt_radius)
    h_melt = float(g.melt_height)

    t_wall = float(g.crucible_wall)
    t_bottom = float(g.crucible_bottom)
    r_crucible_outer = r_melt + t_wall

    heater_gap = float(g.heater_gap)
    heater_thickness = float(g.heater_thickness)
    r_heater_in = r_crucible_outer + heater_gap
    r_heater_out = r_heater_in + heater_thickness
    y_heater_bot = float(g.heater_y_bottom)
    y_heater_top = float(g.heater_y_top)

    r_outer = float(g.outer_radius)
    y_outer_bot = float(g.outer_y_bottom)
    y_outer_top = float(g.outer_y_top)

    y_sl = h_melt
    y_crystal_top = h_melt + h_crystal
    y_argon_top = y_crystal_top + float(g.argon_top_gap)

    # Primary regions
    crystal = Rectangle((0.0, h_melt), (r_crystal, y_crystal_top))
    melt = Rectangle((0.0, 0.0), (r_melt, h_melt))

    crucible_bottom = Rectangle((0.0, -t_bottom), (r_crucible_outer, 0.0))
    crucible_side = Rectangle((r_melt, 0.0), (r_crucible_outer, h_melt))
    crucible = crucible_bottom + crucible_side

    heater = Rectangle((r_heater_in, y_heater_bot), (r_heater_out, y_heater_top))

    outer_box = Rectangle((0.0, y_outer_bot), (r_outer, y_outer_top))
    argon_box = Rectangle((0.0, -t_bottom), (r_heater_in, y_argon_top))

    # Argon cavity around crystal/melt/crucible
    argon = argon_box - crystal - melt - crucible

    # Everything outside argon cavity and outside heater is lumped as insulation/support
    insulation = outer_box - argon_box - heater

    return CzGeometry(
        x=x,
        y=y,
        crystal=crystal,
        melt=melt,
        crucible=crucible,
        argon=argon,
        heater=heater,
        insulation=insulation,
        outer_box=outer_box,
        argon_box=argon_box,
        r_crystal=r_crystal,
        h_crystal=h_crystal,
        r_melt=r_melt,
        h_melt=h_melt,
        t_crucible_wall=t_wall,
        t_crucible_bottom=t_bottom,
        r_crucible_outer=r_crucible_outer,
        heater_gap=heater_gap,
        heater_thickness=heater_thickness,
        r_heater_in=r_heater_in,
        r_heater_out=r_heater_out,
        y_heater_bot=y_heater_bot,
        y_heater_top=y_heater_top,
        r_outer=r_outer,
        y_outer_bot=y_outer_bot,
        y_outer_top=y_outer_top,
        y_sl=y_sl,
        y_crystal_top=y_crystal_top,
        y_argon_top=y_argon_top,
    )