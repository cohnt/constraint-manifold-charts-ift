"""Manim scenes for the RAL supplementary video explanation section.

ProblemScene  (~12s) — the problem, analytic IK parameterization, and
                       why IFT matters for trajectory optimization
PipelineScene (~7s)  — planning pipeline flow chart

Uses bigger fonts scaled down to work around Cairo/Pango small-font spacing bug.

Usage:
    .venv/bin/python -m manim -qh --fps 30 scripts/video/manim_scenes.py ProblemScene PipelineScene
"""

from manim import *

BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT_1 = "#4fc3f7"
ACCENT_2 = "#ff7043"
ACCENT_3 = "#66bb6a"
ACCENT_4 = "#ffd54f"
MANIFOLD_COLOR = "#ab47bc"
SAMPLE_HIT = "#66bb6a"
SAMPLE_MISS = "#ef5350"


def stext(text, font_size=36, **kwargs):
    """Text with bigger internal font, scaled down to avoid spacing bugs."""
    t = Text(text, font_size=font_size * 2, **kwargs)
    t.scale(0.5)
    return t


class ProblemScene(Scene):
    """~14s: problem, IK parameterization, IFT connection to trajopt."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        # --- Part 1: Bimanual constraint ---
        title = stext("Bimanual Manipulation under Constraints", font_size=38, color=TEXT_COLOR)
        subtitle = stext(
            "Two arms must maintain a rigid grasp — a kinematic equality constraint",
            font_size=24, color=ACCENT_1,
        )
        subtitle.next_to(title, DOWN, buff=0.3)
        self.play(FadeIn(title), FadeIn(subtitle), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(title), FadeOut(subtitle), run_time=0.4)

        # --- Part 2: Constraint manifold + sampling failure ---
        cspace_label = stext("Configuration Space (23 DOF)", font_size=26, color=TEXT_COLOR)
        cspace_label.to_edge(UP, buff=0.5)
        rect = Rectangle(width=8, height=4, color=GREY_B, stroke_width=2)
        rect.shift(DOWN * 0.3)

        manifold = ParametricFunction(
            lambda t: np.array([
                t * 3.5 - 3.5,
                0.8 * np.sin(t * 2.5) - 0.3,
                0,
            ]),
            t_range=[0, 2],
            color=MANIFOLD_COLOR,
            stroke_width=4,
        )

        m_label = MathTex(r"\mathcal{M}", font_size=36, color=MANIFOLD_COLOR)
        m_label.move_to(RIGHT * 2.5 + UP * 1.5)
        m_sublabel = stext("constraint manifold (measure zero)", font_size=18, color=MANIFOLD_COLOR)
        m_sublabel.next_to(m_label, RIGHT, buff=0.15)

        self.play(FadeIn(cspace_label), Create(rect), run_time=0.6)
        self.play(Create(manifold), FadeIn(m_label), FadeIn(m_sublabel), run_time=0.8)

        np.random.seed(42)
        samples = []
        for _ in range(30):
            x = np.random.uniform(-3.5, 3.5)
            y = np.random.uniform(-1.7, 1.7) - 0.3
            t_check = (x + 3.5) / 3.5
            near = (0 <= t_check <= 2 and abs(y - (0.8 * np.sin(t_check * 2.5) - 0.3)) < 0.15)
            dot = Dot(point=[x, y, 0], radius=0.06,
                      color=SAMPLE_HIT if near else SAMPLE_MISS)
            dot.set_opacity(0.8)
            samples.append(dot)

        caption1 = stext(
            "Sampling-based planners rarely find feasible configurations",
            font_size=22, color=TEXT_COLOR,
        )
        caption1.to_edge(DOWN, buff=0.4)

        self.play(*[FadeIn(s, scale=0.5) for s in samples], FadeIn(caption1), run_time=0.8)
        self.wait(2.0)

        # --- Part 3: Analytic IK parameterization ---
        self.play(*[FadeOut(s) for s in samples], FadeOut(caption1), FadeOut(cspace_label),
                  run_time=0.4)

        param_label = stext("Analytic IK Parameterization", font_size=28, color=TEXT_COLOR)
        param_label.to_edge(UP, buff=0.5)

        straight_line = Line(
            start=[-3.5, -0.3, 0], end=[3.5, -0.3, 0],
            color=ACCENT_3, stroke_width=4,
        )

        param_eq = MathTex(
            r"\phi : SE(3) \times \psi \;\xrightarrow{\;\text{IK}\;}\; q \in \mathbb{R}^{23}",
            font_size=28, color=ACCENT_1,
        )
        param_eq.move_to(DOWN * 1.8)

        caption2 = stext(
            "IK maps a low-dimensional space directly onto the manifold",
            font_size=22, color=TEXT_COLOR,
        )
        caption2.to_edge(DOWN, buff=0.4)

        param_samples = []
        for i in range(15):
            x = -3.2 + i * (6.4 / 14)
            dot = Dot(point=[x, -0.3, 0], radius=0.06, color=SAMPLE_HIT)
            dot.set_opacity(0.9)
            param_samples.append(dot)

        self.play(
            FadeIn(param_label),
            Transform(manifold, straight_line),
            FadeOut(m_label), FadeOut(m_sublabel), FadeOut(rect),
            run_time=0.8,
        )
        self.play(
            *[FadeIn(s, scale=0.5) for s in param_samples],
            FadeIn(param_eq), FadeIn(caption2),
            run_time=0.8,
        )
        self.wait(2.0)

        # --- Part 4: The IFT connection ---
        self.play(*[FadeOut(s) for s in param_samples],
                  FadeOut(caption2), FadeOut(param_eq),
                  FadeOut(param_label), FadeOut(manifold),
                  run_time=0.4)

        ift_title = stext("The Key Challenge", font_size=32, color=ACCENT_2)
        ift_title.to_edge(UP, buff=0.5)

        # Show the problem: trajopt needs gradients through IK
        box1 = RoundedRectangle(width=3.0, height=0.8, corner_radius=0.1,
                                color=ACCENT_1, stroke_width=2,
                                fill_color=ACCENT_1, fill_opacity=0.15)
        box1.move_to(LEFT * 3.5 + UP * 0.3)
        lbl1 = stext("Analytic IK", font_size=20, color=ACCENT_1)
        lbl1.move_to(box1)

        box2 = RoundedRectangle(width=3.0, height=0.8, corner_radius=0.1,
                                color=ACCENT_2, stroke_width=2,
                                fill_color=ACCENT_2, fill_opacity=0.15)
        box2.move_to(RIGHT * 0 + UP * 0.3)
        lbl2 = stext("Trajectory Optimizer", font_size=20, color=ACCENT_2)
        lbl2.move_to(box2)

        arr1 = Arrow(box1.get_right(), box2.get_left(), buff=0.1,
                     color=GREY_B, stroke_width=2)

        # Gradient arrow going back
        grad_arr = Arrow(box2.get_bottom() + DOWN * 0.3,
                         box1.get_bottom() + DOWN * 0.3,
                         buff=0.1, color=ACCENT_4, stroke_width=3)
        grad_label = MathTex(r"\frac{\partial q}{\partial y}", font_size=24, color=ACCENT_4)
        grad_label.next_to(grad_arr, DOWN, buff=0.15)

        problem_text = stext(
            "Trajopt needs gradients through IK, but IK solvers lack autodiff",
            font_size=20, color=TEXT_COLOR,
        )
        problem_text.move_to(DOWN * 1.5)

        self.play(FadeIn(ift_title), run_time=0.3)
        self.play(FadeIn(VGroup(box1, lbl1)), FadeIn(VGroup(box2, lbl2)),
                  GrowArrow(arr1), run_time=0.6)
        self.play(GrowArrow(grad_arr), FadeIn(grad_label), run_time=0.5)
        self.play(FadeIn(problem_text), run_time=0.4)
        self.wait(1.5)

        # Solution: IFT
        solution = VGroup(
            MathTex(r"D\,\text{FK}_A(q) \cdot \frac{\partial q}{\partial y}"
                    r"= \left(\frac{\partial X}{\partial y},\,"
                    r"\frac{\partial \psi}{\partial y}\right)^T",
                    font_size=26, color=ACCENT_4),
        )
        solution.move_to(DOWN * 1.5)

        solution_caption = stext(
            "IFT: recover gradients from the forward kinematics Jacobian alone",
            font_size=22, color=ACCENT_3,
        )
        solution_caption.to_edge(DOWN, buff=0.4)

        self.play(FadeOut(problem_text), run_time=0.3)
        self.play(Write(solution), run_time=0.8)
        self.play(FadeIn(solution_caption), run_time=0.4)
        self.wait(2.0)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.4)


class PipelineScene(Scene):
    """~7s: planning pipeline."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        pipeline_title = stext("Planning Pipeline", font_size=32, color=TEXT_COLOR)
        pipeline_title.to_edge(UP, buff=0.5)

        stages = [
            ("Goal IK", ACCENT_1),
            ("BiRRT", ACCENT_4),
            ("Shortcut", ACCENT_4),
            ("Trajopt", ACCENT_2),
            ("TOPPRA", ACCENT_3),
        ]

        boxes = []
        arrows = []
        box_width = 1.6
        box_height = 0.7
        gap = 0.5
        total_w = len(stages) * box_width + (len(stages) - 1) * gap
        start_x = -total_w / 2 + box_width / 2

        for i, (name, color) in enumerate(stages):
            x = start_x + i * (box_width + gap)
            box = RoundedRectangle(
                width=box_width, height=box_height,
                corner_radius=0.1, color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.15,
            )
            box.move_to([x, 0, 0])
            label = stext(name, font_size=20, color=color)
            label.move_to(box)
            group = VGroup(box, label)
            boxes.append(group)

            if i > 0:
                prev_box = boxes[i - 1][0]
                arr = Arrow(
                    start=prev_box.get_right(), end=box.get_left(),
                    buff=0.05, color=GREY_B, stroke_width=2,
                    max_tip_length_to_length_ratio=0.3,
                )
                arrows.append(arr)

        # Annotations under key stages
        ann_birrt = stext("sampling-based\npath finding", font_size=14, color=ACCENT_4)
        ann_birrt.next_to(boxes[1], DOWN, buff=0.4)

        ann_trajopt = stext("gradient-based\nsmoothing (IFT)", font_size=14, color=ACCENT_2)
        ann_trajopt.next_to(boxes[3], DOWN, buff=0.4)

        ann_toppra = stext("time-optimal\nretiming", font_size=14, color=ACCENT_3)
        ann_toppra.next_to(boxes[4], DOWN, buff=0.4)

        self.play(FadeIn(pipeline_title), run_time=0.3)

        for i, group in enumerate(boxes):
            anims = [FadeIn(group, shift=UP * 0.2)]
            if i > 0:
                anims.append(GrowArrow(arrows[i - 1]))
            self.play(*anims, run_time=0.3)

        self.play(FadeIn(ann_birrt), FadeIn(ann_trajopt), FadeIn(ann_toppra), run_time=0.5)

        caption = stext(
            "IFT enables gradient-based trajectory optimization through black-box IK solvers",
            font_size=22, color=TEXT_COLOR,
        )
        caption.to_edge(DOWN, buff=0.4)
        self.play(FadeIn(caption), run_time=0.4)
        self.wait(1.5)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.4)
