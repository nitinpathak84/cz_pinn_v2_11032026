from sympy import Eq, And, Or

from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer


def add_boundary_constraints(domain, nodes, geo, cfg):
    x = geo.x
    y = geo.y
    bs = cfg.batch_size
    bc = cfg.custom.boundary

    theta_seed = float(bc.theta_seed)
    theta_hot = float(bc.theta_hot)
    theta_sl = float(bc.theta_sl)

    # ---------------------------------------------------------
    # Axis symmetry
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"normal_gradient_theta_cr": 0.0},
            batch_size=bs.axis_cr,
            criteria=Eq(x, 0.0),
        ),
        "axis_crystal",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"normal_gradient_theta_m": 0.0},
            batch_size=bs.axis_m,
            criteria=Eq(x, 0.0),
        ),
        "axis_melt",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={"normal_gradient_theta_cu": 0.0},
            batch_size=bs.axis_cu,
            criteria=Eq(x, 0.0),
        ),
        "axis_crucible",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.argon,
            outvar={"normal_gradient_theta_ar": 0.0},
            batch_size=bs.axis_ar,
            criteria=Eq(x, 0.0),
        ),
        "axis_argon",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.axis_ins,
            criteria=Eq(x, 0.0),
        ),
        "axis_insulation",
    )

    # ---------------------------------------------------------
    # Crystal top
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"theta_cr": theta_seed},
            batch_size=bs.crystal_top,
            criteria=Eq(y, geo.y_crystal_top),
        ),
        "crystal_top",
    )

    # ---------------------------------------------------------
    # Heater driven boundary: outer radial face
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.heater,
            outvar={"theta_ht": theta_hot},
            batch_size=bs.heater_outer,
            criteria=Eq(x, geo.r_heater_out),
        ),
        "heater_outer_face",
    )

    # ---------------------------------------------------------
    # Outer insulation boundaries: adiabatic for V2
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.outer_right,
            criteria=Eq(x, geo.r_outer),
        ),
        "outer_right_adiabatic",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.outer_top,
            criteria=Eq(y, geo.y_outer_top),
        ),
        "outer_top_adiabatic",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.outer_bottom,
            criteria=Eq(y, geo.y_outer_bot),
        ),
        "outer_bottom_adiabatic",
    )

    # ---------------------------------------------------------
    # Fixed solid-liquid interface temperature
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"theta_cr": theta_sl},
            batch_size=bs.sl_cr,
            criteria=And(Eq(y, geo.y_sl), x <= geo.r_crystal),
        ),
        "sl_interface_crystal_side",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"theta_m": theta_sl},
            batch_size=bs.sl_m,
            criteria=And(Eq(y, geo.y_sl), x <= geo.r_crystal),
        ),
        "sl_interface_melt_side",
    )


def add_interior_constraints(domain, nodes, geo, cfg):
    bs = cfg.batch_size

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"axisym_diffusion_theta_cr": 0.0},
            batch_size=bs.interior_cr,
        ),
        "interior_crystal",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"axisym_diffusion_theta_m": 0.0},
            batch_size=bs.interior_m,
        ),
        "interior_melt",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={"axisym_diffusion_theta_cu": 0.0},
            batch_size=bs.interior_cu,
        ),
        "interior_crucible",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.argon,
            outvar={"axisym_diffusion_theta_ar": 0.0},
            batch_size=bs.interior_ar,
        ),
        "interior_argon",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.heater,
            outvar={"axisym_diffusion_theta_ht": 0.0},
            batch_size=bs.interior_ht,
        ),
        "interior_heater",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"axisym_diffusion_theta_ins": 0.0},
            batch_size=bs.interior_ins,
        ),
        "interior_insulation",
    )


def add_interface_constraints(domain, nodes, geo, cfg):
    x = geo.x
    y = geo.y
    bs = cfg.batch_size

    # ---------------------------------------------------------
    # Melt <-> Crucible
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={
                "diffusion_interface_dirichlet_theta_m_theta_cu": 0.0,
                "diffusion_interface_neumann_theta_m_theta_cu": 0.0,
            },
            batch_size=bs.interface_mc,
            criteria=Or(Eq(y, 0.0), Eq(x, geo.r_melt)),
        ),
        "interface_melt_crucible",
    )

    # ---------------------------------------------------------
    # Crucible <-> Argon
    # outer side wall + top lip outside melt footprint
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={
                "diffusion_interface_dirichlet_theta_cu_theta_ar": 0.0,
                "diffusion_interface_neumann_theta_cu_theta_ar": 0.0,
            },
            batch_size=bs.interface_ca,
            criteria=Or(
                Eq(x, geo.r_crucible_outer),
                And(Eq(y, geo.h_melt), x >= geo.r_melt),
            ),
        ),
        "interface_crucible_argon",
    )

    # ---------------------------------------------------------
    # Crystal <-> Argon
    # crystal side wall only
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={
                "diffusion_interface_dirichlet_theta_cr_theta_ar": 0.0,
                "diffusion_interface_neumann_theta_cr_theta_ar": 0.0,
            },
            batch_size=bs.interface_cr_ar,
            criteria=Eq(x, geo.r_crystal),
        ),
        "interface_crystal_argon",
    )

    # ---------------------------------------------------------
    # Melt <-> Argon
    # melt free surface outside crystal footprint
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={
                "diffusion_interface_dirichlet_theta_m_theta_ar": 0.0,
                "diffusion_interface_neumann_theta_m_theta_ar": 0.0,
            },
            batch_size=bs.interface_m_ar,
            criteria=And(Eq(y, geo.h_melt), x >= geo.r_crystal),
        ),
        "interface_melt_argon",
    )

    # ---------------------------------------------------------
    # Argon <-> Heater
    # inner heater face
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.argon,
            outvar={
                "diffusion_interface_dirichlet_theta_ar_theta_ht": 0.0,
                "diffusion_interface_neumann_theta_ar_theta_ht": 0.0,
            },
            batch_size=bs.interface_ar_ht,
            criteria=And(
                Eq(x, geo.r_heater_in),
                y >= geo.y_heater_bot,
                y <= geo.y_heater_top,
            ),
        ),
        "interface_argon_heater",
    )

    # ---------------------------------------------------------
    # Argon <-> Insulation
    # top of argon cavity + right side where heater is absent
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.argon,
            outvar={
                "diffusion_interface_dirichlet_theta_ar_theta_ins": 0.0,
                "diffusion_interface_neumann_theta_ar_theta_ins": 0.0,
            },
            batch_size=bs.interface_ar_ins,
            criteria=Or(
                Eq(y, geo.y_argon_top),
                And(
                    Eq(x, geo.r_heater_in),
                    Or(y < geo.y_heater_bot, y > geo.y_heater_top),
                ),
            ),
        ),
        "interface_argon_insulation",
    )

    # ---------------------------------------------------------
    # Heater <-> Insulation
    # heater top and bottom faces only
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.heater,
            outvar={
                "diffusion_interface_dirichlet_theta_ht_theta_ins": 0.0,
                "diffusion_interface_neumann_theta_ht_theta_ins": 0.0,
            },
            batch_size=bs.interface_ht_ins,
            criteria=Or(Eq(y, geo.y_heater_bot), Eq(y, geo.y_heater_top)),
        ),
        "interface_heater_insulation",
    )

    # ---------------------------------------------------------
    # Crucible <-> Insulation
    # crucible bottom outer face
    # ---------------------------------------------------------
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={
                "diffusion_interface_dirichlet_theta_cu_theta_ins": 0.0,
                "diffusion_interface_neumann_theta_cu_theta_ins": 0.0,
            },
            batch_size=bs.interface_cu_ins,
            criteria=Eq(y, -geo.t_crucible_bottom),
        ),
        "interface_crucible_insulation",
    )


def add_inferencers(domain, nodes, geo, cfg):
    inf = cfg.custom.inference
    bs = int(inf.batch_size)

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=geo.crystal.sample_interior(nr_points=int(inf.n_crystal)),
            output_names=["theta_cr"],
            batch_size=bs,
        ),
        "crystal",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=geo.melt.sample_interior(nr_points=int(inf.n_melt)),
            output_names=["theta_m"],
            batch_size=bs,
        ),
        "melt",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=geo.crucible.sample_interior(nr_points=int(inf.n_crucible)),
            output_names=["theta_cu"],
            batch_size=bs,
        ),
        "crucible",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=geo.argon.sample_interior(nr_points=int(inf.n_argon)),
            output_names=["theta_ar"],
            batch_size=bs,
        ),
        "argon",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=geo.heater.sample_interior(nr_points=int(inf.n_heater)),
            output_names=["theta_ht"],
            batch_size=bs,
        ),
        "heater",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=geo.insulation.sample_interior(nr_points=int(inf.n_insulation)),
            output_names=["theta_ins"],
            batch_size=bs,
        ),
        "insulation",
    )


def add_monitors(domain, nodes, geo, cfg):
    return