from dataclasses import dataclass, field


@dataclass
class PhysicsConfig:
    ticks_per_sec: int = 20
    gravity: float = 28.0
    terminal_velocity: float = -50.0
    jump_velocity: float = 8.8
    walk_speed: float = 4.3
    sprint_speed: float = 5.6
    ground_accel: float = 12.0
    air_accel: float = 2.5
    player_height: float = 1.8
    player_radius: float = 0.3

    @property
    def dt(self) -> float:
        return 1.0 / float(self.ticks_per_sec)


@dataclass
class CombatConfig:
    max_health: float = 20.0
    attack_range: float = 3.0
    attack_arc_deg: float = 75.0
    # vertical tolerance, split from the horizontal cone. When equal to
    # attack_arc_deg the original single-cone check runs bit-identically.
    attack_arc_vertical_deg: float = 75.0
    cooldown_ticks: int = 12
    hurt_ticks: int = 10
    base_damage: float = 5.0
    crit_mult: float = 1.5
    knockback: float = 0.45
    knockback_sprint_bonus: float = 0.25
    knockback_up: float = 0.38
    # attack lockout right after landing (recovery). 0 = baseline behavior.
    landing_attack_delay_ticks: int = 0

    @property
    def mechanics_id(self) -> str:
        return (f"crit{self.crit_mult}_arcH{self.attack_arc_deg}_arcV"
                f"{self.attack_arc_vertical_deg}_land{self.landing_attack_delay_ticks}")


@dataclass
class WorldConfig:
    size: int = 20
    wall_margin: float = 1.0
    # obstacles as (x0, z0, x1, z1, height); deterministic defaults, can be []
    obstacles: tuple = ((8.0, 8.0, 9.0, 9.0, 2.0), (11.0, 11.0, 12.0, 12.0, 2.0))

    def without_obstacles(self) -> "WorldConfig":
        return WorldConfig(size=self.size, wall_margin=self.wall_margin, obstacles=())


@dataclass
class SimConfig:
    max_ticks: int = 900  # 45s at 20 TPS
    frame_skip: int = 4   # physics ticks per policy decision
    seed: int = 0
