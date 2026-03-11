from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.basic import GradNormal
from physicsnemo.sym.eq.pdes.diffusion import DiffusionInterface

from cz.pdes.axisymmetric_diffusion import AxisymmetricDiffusion


def build_cz_nodes(cfg):
    aspect_sq = float(cfg.custom.nondim.aspect_sq)
    eps_r = float(cfg.custom.numerics.eps_r)

    # Interior PDEs
    eq_cr = AxisymmetricDiffusion(T="theta_cr", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_m = AxisymmetricDiffusion(T="theta_m", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_cu = AxisymmetricDiffusion(T="theta_cu", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_ar = AxisymmetricDiffusion(T="theta_ar", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_ht = AxisymmetricDiffusion(T="theta_ht", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_ins = AxisymmetricDiffusion(T="theta_ins", aspect_sq=aspect_sq, eps_r=eps_r)

    # Normal-gradient nodes
    gn_cr = GradNormal(T="theta_cr", dim=2, time=False)
    gn_m = GradNormal(T="theta_m", dim=2, time=False)
    gn_cu = GradNormal(T="theta_cu", dim=2, time=False)
    gn_ar = GradNormal(T="theta_ar", dim=2, time=False)
    gn_ht = GradNormal(T="theta_ht", dim=2, time=False)
    gn_ins = GradNormal(T="theta_ins", dim=2, time=False)

    # Interface nodes
    if_mc = DiffusionInterface(
        "theta_m",
        "theta_cu",
        float(cfg.custom.physics.k_m),
        float(cfg.custom.physics.k_cu),
        dim=2,
        time=False,
    )

    if_ca = DiffusionInterface(
        "theta_cu",
        "theta_ar",
        float(cfg.custom.physics.k_cu),
        float(cfg.custom.physics.k_ar),
        dim=2,
        time=False,
    )

    if_cr_ar = DiffusionInterface(
        "theta_cr",
        "theta_ar",
        float(cfg.custom.physics.k_cr),
        float(cfg.custom.physics.k_ar),
        dim=2,
        time=False,
    )

    if_m_ar = DiffusionInterface(
        "theta_m",
        "theta_ar",
        float(cfg.custom.physics.k_m),
        float(cfg.custom.physics.k_ar),
        dim=2,
        time=False,
    )

    if_ar_ht = DiffusionInterface(
        "theta_ar",
        "theta_ht",
        float(cfg.custom.physics.k_ar),
        float(cfg.custom.physics.k_ht),
        dim=2,
        time=False,
    )

    if_ar_ins = DiffusionInterface(
        "theta_ar",
        "theta_ins",
        float(cfg.custom.physics.k_ar),
        float(cfg.custom.physics.k_ins),
        dim=2,
        time=False,
    )

    if_ht_ins = DiffusionInterface(
        "theta_ht",
        "theta_ins",
        float(cfg.custom.physics.k_ht),
        float(cfg.custom.physics.k_ins),
        dim=2,
        time=False,
    )

    if_cu_ins = DiffusionInterface(
        "theta_cu",
        "theta_ins",
        float(cfg.custom.physics.k_cu),
        float(cfg.custom.physics.k_ins),
        dim=2,
        time=False,
    )

    input_keys = [Key("x"), Key("y")]

    net_cr = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_cr")],
        cfg=cfg.arch.fully_connected,
    )
    net_m = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_m")],
        cfg=cfg.arch.fully_connected,
    )
    net_cu = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_cu")],
        cfg=cfg.arch.fully_connected,
    )
    net_ar = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_ar")],
        cfg=cfg.arch.fully_connected,
    )
    net_ht = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_ht")],
        cfg=cfg.arch.fully_connected,
    )
    net_ins = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_ins")],
        cfg=cfg.arch.fully_connected,
    )

    nodes = (
        eq_cr.make_nodes()
        + eq_m.make_nodes()
        + eq_cu.make_nodes()
        + eq_ar.make_nodes()
        + eq_ht.make_nodes()
        + eq_ins.make_nodes()
        + gn_cr.make_nodes()
        + gn_m.make_nodes()
        + gn_cu.make_nodes()
        + gn_ar.make_nodes()
        + gn_ht.make_nodes()
        + gn_ins.make_nodes()
        + if_mc.make_nodes()
        + if_ca.make_nodes()
        + if_cr_ar.make_nodes()
        + if_m_ar.make_nodes()
        + if_ar_ht.make_nodes()
        + if_ar_ins.make_nodes()
        + if_ht_ins.make_nodes()
        + if_cu_ins.make_nodes()
        + [net_cr.make_node(name="theta_cr_net")]
        + [net_m.make_node(name="theta_m_net")]
        + [net_cu.make_node(name="theta_cu_net")]
        + [net_ar.make_node(name="theta_ar_net")]
        + [net_ht.make_node(name="theta_ht_net")]
        + [net_ins.make_node(name="theta_ins_net")]
    )

    return nodes