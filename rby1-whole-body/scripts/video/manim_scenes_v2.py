"""Manim scenes for the promotional video (Video 2).

IKParameterizationScene (~55s):
  Act 1: The problem — sampling failure + projection sampler visualization
  Act 2: IK parameterization — flattening with lifting arrows and particle traces
  Act 3: The IFT — commutative diagram and gradient equation

RBY1PipelineScene (~8s): RBY1-specific planning pipeline, shown before the RBY1 section.

Usage:
    .venv/bin/python -m manim -qh --fps 30 scripts/video/manim_scenes_v2.py IKParameterizationScene
    .venv/bin/python -m manim -qh --fps 30 scripts/video/manim_scenes_v2.py RBY1PipelineScene
"""

from manim import *

BG_COLOR = "#1a1a2e"
TEXT_COLOR = "#e0e0e0"
ACCENT_1 = "#4fc3f7"   # blue
ACCENT_2 = "#ff7043"   # coral
ACCENT_3 = "#66bb6a"   # green
ACCENT_4 = "#ffd54f"   # yellow
MANIFOLD_COLOR = "#ab47bc"
SAMPLE_HIT = "#66bb6a"
SAMPLE_MISS = "#ef5350"


def stext(text, font_size=36, line_spacing=1.15, **kwargs):
    # Rendered at 2x and scaled down to dodge the Cairo/Pango spacing bug at
    # small font sizes. line_spacing keeps the descenders of one line clear of
    # the ascenders of the next in the small multi-line captions.
    t = Text(text, font_size=font_size * 2, line_spacing=line_spacing, **kwargs)
    t.scale(0.5)
    return t


def manifold_y(x):
    t = (x + 3.5) / 3.5
    if 0 <= t <= 2:
        return 0.8 * np.sin(t * 2.5) - 0.3
    return -0.3


def closest_on_manifold(px, py):
    best_x, best_d = px, 1e9
    for t in np.linspace(0, 2, 200):
        mx = t * 3.5 - 3.5
        my = 0.8 * np.sin(t * 2.5) - 0.3
        d = (px - mx)**2 + (py - my)**2
        if d < best_d:
            best_d = d
            best_x = mx
    return best_x, manifold_y(best_x)


class IKParameterizationScene(Scene):

    def construct(self):
        self.camera.background_color = BG_COLOR

        # ======================== ACT 1: THE PROBLEM ========================

        title = stext("Planning Under Equality Constraints", font_size=34, color=TEXT_COLOR)
        title.to_edge(UP, buff=0.5)

        rect = Rectangle(width=8, height=4, color=GREY_B, stroke_width=2)
        rect.shift(DOWN * 0.3)

        cspace_label = stext("Configuration Space (n DOF)", font_size=22, color=GREY_B)
        cspace_label.next_to(rect, UP, buff=0.1)

        manifold = ParametricFunction(
            lambda t: np.array([t * 3.5 - 3.5, 0.8 * np.sin(t * 2.5) - 0.3, 0]),
            t_range=[0, 2], color=MANIFOLD_COLOR, stroke_width=4,
        )
        # The label lives in the gutter right of the rectangle (which ends at
        # x=4.0). m_sub is 2.07 wide, so centring it at x=5.45 clears the border
        # by 0.4 and the frame edge by 0.6 -- at x=5.0 it crossed the border.
        m_label = MathTex(r"\mathcal{M}", font_size=36, color=MANIFOLD_COLOR)
        m_label.move_to(RIGHT * 5.45 + UP * 0.35)
        m_sub = stext("constraint manifold\n(measure zero)", font_size=16, color=MANIFOLD_COLOR)
        m_sub.next_to(m_label, DOWN, buff=0.15)

        self.play(FadeIn(title), Create(rect), FadeIn(cspace_label), run_time=1.0)
        self.wait(0.5)
        self.play(Create(manifold), FadeIn(m_label), FadeIn(m_sub), run_time=1.2)
        self.wait(1.0)

        # Explanatory text
        explain1 = stext(
            "The feasible set is a thin surface in a high-dimensional space",
            font_size=18, color=ACCENT_1,
        )
        explain1.to_edge(DOWN, buff=0.6)
        self.play(FadeIn(explain1), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(explain1), run_time=0.5)

        # Random samples — ALL miss (reject any near the manifold)
        np.random.seed(42)
        samples = []
        sample_positions = []
        while len(samples) < 35:
            x = np.random.uniform(-3.3, 3.3)
            y = np.random.uniform(-1.5, 1.5) - 0.3
            _, my = closest_on_manifold(x, y)
            if abs(y - my) < 0.15:
                continue
            dot = Dot(point=[x, y, 0], radius=0.055, color=SAMPLE_MISS)
            dot.set_opacity(0.8)
            samples.append(dot)
            sample_positions.append((x, y))

        caption1 = stext(
            "Sampling-based planners never find feasible configurations\n"
            "on a measure-zero manifold",
            font_size=20, color=TEXT_COLOR,
        )
        caption1.to_edge(DOWN, buff=0.3)

        self.play(*[FadeIn(s, scale=0.5) for s in samples], FadeIn(caption1), run_time=1.2)
        self.wait(3.0)

        # Projection sampling visualization
        self.play(FadeOut(caption1), run_time=0.5)

        caption_proj = stext(
            "Projection-based samplers can find feasible samples,\n"
            "but projections are expensive and may fail",
            font_size=20, color=TEXT_COLOR,
        )
        caption_proj.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(caption_proj), run_time=0.5)

        animations = []
        for i, (dot, (px, py)) in enumerate(zip(samples, sample_positions)):
            mx, my = closest_on_manifold(px, py)
            dist = np.sqrt((px - mx)**2 + (py - my)**2)
            if dist < 1.5:
                animations.append(dot.animate.move_to([mx, my, 0]).set_color(SAMPLE_HIT))
            else:
                mid_x = px + (mx - px) * 0.4
                mid_y = py + (my - py) * 0.4
                animations.append(dot.animate.move_to([mid_x, mid_y, 0]).set_opacity(0.3))

        self.play(*animations, run_time=2.5)
        self.wait(2.5)

        self.play(
            *[FadeOut(s) for s in samples],
            FadeOut(caption_proj),
            run_time=0.6,
        )

        # ======================== ACT 2: PARAMETERIZATION ========================

        # The general idea first, the specific construction second. Analytic IK
        # is *a* way to build a parameterization of these manifolds, not what a
        # parameterization is -- titling this act "Analytic IK Parameterization"
        # conflated the two.
        param_title = stext("Parameterizing the Constraint Manifold",
                            font_size=28, color=TEXT_COLOR)
        param_title.to_edge(UP, buff=0.5)
        self.play(FadeOut(title), FadeOut(cspace_label), FadeIn(param_title), run_time=0.8)

        param_line = Line(
            start=[-3.5, -2.8, 0], end=[3.5, -2.8, 0],
            color=ACCENT_3, stroke_width=4,
        )
        param_line_label = stext("Parameterized Space (m DOF)", font_size=18, color=ACCENT_3)
        # Above the line, in the gap between it and the rectangle's bottom edge
        # (y=-2.30). Below the line it was 0.1 off the stroke and left too
        # little room for the two-line captions that share the bottom band.
        param_line_label.next_to(param_line, UP, buff=0.18)

        self.play(Create(param_line), FadeIn(param_line_label), run_time=1.0)

        manifold_copy = manifold.copy()
        self.add(manifold_copy)

        flat_line = ParametricFunction(
            lambda t: np.array([t * 3.5 - 3.5, -2.8, 0]),
            t_range=[0, 2], color=ACCENT_3, stroke_width=3,
        )
        flat_line.set_opacity(0.5)

        self.play(Transform(manifold_copy, flat_line), run_time=2.0)
        self.wait(1.0)

        # Particle traces: animate dots sweeping along parameterized line
        # with traced paths on the manifold.
        #
        # These traces are *recomputed* from each dot's current position every
        # frame (swept_arc / swept_segment below), not accumulated from the
        # dot's history. That distinction is load-bearing, and manim's
        # partial-movie-file cache is why:
        #
        #   TracedPath records one point per rendered frame. When manim gets a
        #   cache hit on an animation it does not step that animation -- it
        #   fast-forwards it in a single time step and replays the stored
        #   frames. A TracedPath spanning a cached animation therefore ends up
        #   with exactly one segment, start straight to end. The cached frames
        #   still show the correct curve, so the animation itself looks right;
        #   every *later* animation, rendered fresh, draws the degenerate
        #   straight chord instead. That is a silent regression that appears
        #   only after an edit downstream of the sweep, and it shipped in
        #   overview_video.mp4 once (the mixed Aug/Sep partial_movie_files under
        #   media/videos/manim_scenes_v2/ are the fingerprint).
        #
        # A trace that is a pure function of the dot's current position has no
        # history to lose, so it renders identically whether the sweep was
        # stepped frame by frame or fast-forwarded in one jump.
        n_particles = 6
        trace_dots = []
        traced_paths = []
        manifold_trace_dots = []
        manifold_traced_paths = []

        # always_redraw drives its mobject with become(), and become() cannot
        # repopulate a VMobject that was constructed empty -- it silently stays
        # empty for the rest of the scene. Both traces are therefore built with
        # a fixed point count at every span, including the zero span they have
        # before the sweep starts, and hidden by opacity rather than by having
        # no geometry.
        TRACE_SAMPLES = 64
        MIN_SPAN = 1e-3

        def swept_arc(pd, x0):
            """The piece of the manifold between x0 and pd's current x."""
            def redraw():
                x1 = pd.get_center()[0]
                span = abs(x1 - x0)
                lo, hi = min(x0, x1), max(x0, x1)
                if span < MIN_SPAN:
                    hi = lo + MIN_SPAN
                arc = VMobject()
                arc.set_points_smoothly([
                    np.array([x, manifold_y(x), 0])
                    for x in np.linspace(lo, hi, TRACE_SAMPLES)
                ])
                arc.set_stroke(color=ACCENT_4, width=2.5,
                               opacity=0.6 if span >= MIN_SPAN else 0.0)
                return arc
            return always_redraw(redraw)

        def swept_segment(pd, x0):
            """The piece of the parameterized line between x0 and pd's x."""
            def redraw():
                x1 = pd.get_center()[0]
                span = abs(x1 - x0)
                lo, hi = min(x0, x1), max(x0, x1)
                if span < MIN_SPAN:
                    hi = lo + MIN_SPAN
                seg = VMobject()
                seg.set_points_smoothly([
                    np.array([x, -2.8, 0])
                    for x in np.linspace(lo, hi, TRACE_SAMPLES)
                ])
                seg.set_stroke(color=ACCENT_3, width=2.5,
                               opacity=0.5 if span >= MIN_SPAN else 0.0)
                return seg
            return always_redraw(redraw)

        for i in range(n_particles):
            start_x = -3.2 + i * (6.4 / (n_particles - 1))
            p_dot = Dot(point=[start_x, -2.8, 0], radius=0.07, color=ACCENT_3)
            p_dot.set_opacity(0.9)
            trace_dots.append(p_dot)

            traced_paths.append(swept_segment(p_dot, start_x))

            my = manifold_y(start_x)
            m_dot = Dot(point=[start_x, my, 0], radius=0.07, color=ACCENT_4)
            m_dot.set_opacity(0.9)
            manifold_trace_dots.append(m_dot)

            manifold_traced_paths.append(swept_arc(p_dot, start_x))

        for d in trace_dots + manifold_trace_dots:
            self.add(d)
        for tp in traced_paths + manifold_traced_paths:
            self.add(tp)

        explain_param = stext(
            "A parameterization maps each low-dimensional parameter\n"
            "to a feasible configuration on the manifold",
            font_size=18, color=TEXT_COLOR,
        )
        explain_param.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(explain_param), run_time=0.5)

        sweep_anims = []
        for i, (p_dot, m_dot) in enumerate(zip(trace_dots, manifold_trace_dots)):
            end_x = -3.2 + i * (6.4 / (n_particles - 1))
            target_x = min(end_x + 1.5, 3.3)

            def make_m_updater(pd, md):
                def updater(mob):
                    x = pd.get_center()[0]
                    y = manifold_y(x)
                    mob.move_to([x, y, 0])
                return updater

            m_dot.add_updater(make_m_updater(p_dot, m_dot))
            sweep_anims.append(p_dot.animate.move_to([target_x, -2.8, 0]))

        self.play(*sweep_anims, run_time=3.0, rate_func=smooth)

        # Freeze the traces at their final geometry. update() forces one last
        # redraw at the end pose so the arc is complete no matter how the sweep
        # was rendered; clear_updaters() then stops always_redraw from
        # rebuilding -- and resetting the opacity of -- these mobjects, which is
        # what lets the Act 2 FadeOut below actually fade them.
        for tp in traced_paths + manifold_traced_paths:
            tp.update()
            tp.clear_updaters()

        for m_dot in manifold_trace_dots:
            m_dot.clear_updaters()

        self.wait(1.0)

        # Beat 1: the general object -- any chart from a low-dimensional
        # parameter space onto the manifold.
        param_eq = MathTex(
            r"\phi : \mathcal{Y} \;\longrightarrow\; \mathcal{M} \subset \mathbb{R}^n",
            font_size=28, color=ACCENT_1,
        )
        # Sits in the band between the title and the top of the rectangle
        # (y=1.70), which cspace_label vacated at the start of this act. At
        # RIGHT*4.5 + DOWN*2.0 it straddled the rectangle's right border.
        param_eq.move_to(UP * 2.35)

        self.play(FadeOut(explain_param), run_time=0.3)
        caption2 = stext(
            "Planning in the parameter space satisfies the constraint by construction",
            font_size=20, color=TEXT_COLOR,
        )
        caption2.to_edge(DOWN, buff=0.3)

        self.play(FadeIn(param_eq), FadeIn(caption2), run_time=0.8)
        self.wait(2.5)

        # Beat 2: analytic IK as one construction of that map, for the
        # manifolds this paper cares about. Replacing the general equation with
        # the concrete one, rather than showing only the concrete one, is the
        # whole point of splitting this into two beats.
        param_eq2 = MathTex(
            r"\phi : SE(3) \times \psi \;\xrightarrow{\;\text{analytic IK}\;}\; "
            r"q \in \mathbb{R}^n",
            font_size=28, color=ACCENT_3,
        )
        param_eq2.move_to(param_eq)

        caption3 = stext(
            "Analytic IK is one way to construct such a parameterization,\n"
            "for the constraint manifolds we care about",
            font_size=20, color=ACCENT_3,
        )
        caption3.to_edge(DOWN, buff=0.3)

        self.play(FadeOut(caption2), run_time=0.3)
        self.play(Transform(param_eq, param_eq2), FadeIn(caption3), run_time=0.9)
        self.wait(3.0)

        # Clear for Act 3
        all_act2 = [
            *trace_dots, *manifold_trace_dots, *traced_paths, *manifold_traced_paths,
            manifold, manifold_copy, rect, m_label, m_sub,
            param_line, param_line_label, param_eq, caption3, param_title,
        ]
        self.play(*[FadeOut(m) for m in all_act2], run_time=0.8)

        # ======================== ACT 3: THE IFT ========================

        # Sub-part 3a: The problem
        ift_title = stext("The Key Challenge: Gradients", font_size=30, color=ACCENT_2)
        ift_title.to_edge(UP, buff=0.5)

        # The whole evaluation chain lives INSIDE the trajectory optimizer:
        # decision variables -> analytic IK -> costs and constraints. Drawing
        # the optimizer as a peer box beside the IK (as this once did) implied
        # the IK was something the optimizer called out to, when in fact it sits
        # on the optimizer's own inner loop -- which is exactly why its missing
        # derivative is fatal.
        umbrella = RoundedRectangle(width=12.6, height=3.7, corner_radius=0.18,
                                    color=GREY_B, stroke_width=2,
                                    fill_color=GREY_B, fill_opacity=0.06)
        umbrella.move_to(UP * 0.5)
        umbrella_lbl = stext("Trajectory Optimizer", font_size=20, color=GREY_B)
        umbrella_lbl.move_to(umbrella.get_top() + DOWN * 0.35)

        BOX_W, BOX_H, BOX_Y = 3.5, 1.15, 0.95

        def stage(x, label, color, font_size=19):
            box = RoundedRectangle(width=BOX_W, height=BOX_H, corner_radius=0.1,
                                   color=color, stroke_width=2,
                                   fill_color=color, fill_opacity=0.15)
            box.move_to(RIGHT * x + UP * BOX_Y)
            lbl = stext(label, font_size=font_size, color=color)
            lbl.move_to(box)
            return box, lbl

        box_var, lbl_var = stage(-4.15, "Decision\nVariables", ACCENT_1)
        box_ik, lbl_ik = stage(0.0, "Analytic IK", ACCENT_4, font_size=20)
        box_cost, lbl_cost = stage(4.15, "Costs and\nConstraints", ACCENT_2)

        fwd1 = Arrow(box_var.get_right(), box_ik.get_left(), buff=0.12,
                     color=GREY_B, stroke_width=3)
        fwd2 = Arrow(box_ik.get_right(), box_cost.get_left(), buff=0.12,
                     color=GREY_B, stroke_width=3)

        # The gradient path runs back along the same chain, inside the umbrella,
        # and is broken at the IK stage -- the cross sits under box_ik, naming
        # exactly which link is missing rather than the return path in general.
        grad_y = -0.75
        grad_arrow = Arrow(RIGHT * 4.15 + UP * grad_y, RIGHT * -4.15 + UP * grad_y,
                           buff=0.0, color=ACCENT_4, stroke_width=3)
        drop_r = DashedLine(box_cost.get_bottom(), RIGHT * 4.15 + UP * grad_y,
                            color=ACCENT_4, stroke_width=2, dash_length=0.12)
        rise_l = DashedLine(RIGHT * -4.15 + UP * grad_y, box_var.get_bottom(),
                            color=ACCENT_4, stroke_width=2, dash_length=0.12)

        block_x = Cross(stroke_color=SAMPLE_MISS, stroke_width=6)
        block_x.scale(0.28)
        block_x.move_to(UP * grad_y)

        # Above the return arrow, not below it: the umbrella's bottom edge is at
        # y=-1.35 and this fraction is 0.62 tall, so hanging it under an arrow at
        # y=-0.75 puts it across that border.
        grad_lbl = MathTex(r"\frac{\partial q}{\partial y}", font_size=26,
                           color=ACCENT_4)
        grad_lbl.move_to(RIGHT * 2.1 + UP * (grad_y + 0.45))

        problem_cap = stext(
            "The optimizer needs derivatives through every stage,\n"
            "but an analytic IK solver provides none",
            font_size=20, color=TEXT_COLOR,
        )
        problem_cap.move_to(DOWN * 2.85)

        self.play(FadeIn(ift_title), run_time=0.5)
        self.play(Create(umbrella), FadeIn(umbrella_lbl), run_time=0.6)
        self.play(FadeIn(VGroup(box_var, lbl_var)), run_time=0.4)
        self.play(GrowArrow(fwd1), FadeIn(VGroup(box_ik, lbl_ik)), run_time=0.5)
        self.play(GrowArrow(fwd2), FadeIn(VGroup(box_cost, lbl_cost)), run_time=0.5)
        self.wait(0.6)
        self.play(Create(drop_r), GrowArrow(grad_arrow), Create(rise_l),
                  FadeIn(grad_lbl), run_time=1.0)
        self.play(Create(block_x), FadeIn(problem_cap), run_time=0.6)
        self.wait(3.0)

        # Sub-part 3b: IFT solution — commutative diagram
        self.play(
            *[FadeOut(m) for m in [umbrella, umbrella_lbl,
                                    box_var, lbl_var, box_ik, lbl_ik,
                                    box_cost, lbl_cost, fwd1, fwd2,
                                    drop_r, grad_arrow, rise_l, grad_lbl,
                                    block_x, problem_cap, ift_title]],
            run_time=0.6,
        )

        ift_title2 = stext("The Inverse Function Theorem", font_size=30, color=ACCENT_3)
        ift_title2.to_edge(UP, buff=0.5)

        node_y = MathTex(r"y", font_size=36, color=ACCENT_1)
        node_y.move_to(LEFT * 2.5 + UP * 1.0)
        node_q = MathTex(r"q", font_size=36, color=ACCENT_2)
        node_q.move_to(RIGHT * 2.5 + UP * 1.0)

        arrow_ik = Arrow(node_y.get_right() + RIGHT * 0.2,
                         node_q.get_left() + LEFT * 0.2,
                         buff=0.1, color=GREY_B, stroke_width=2)
        ik_label = MathTex(r"\text{IK}", font_size=24, color=GREY_B)
        ik_label.next_to(arrow_ik, UP, buff=0.1)

        arrow_fk = Arrow(node_q.get_left() + LEFT * 0.2 + DOWN * 0.5,
                         node_y.get_right() + RIGHT * 0.2 + DOWN * 0.5,
                         buff=0.1, color=ACCENT_3, stroke_width=2)
        fk_label = MathTex(r"\text{FK}_A", font_size=24, color=ACCENT_3)
        fk_label.next_to(arrow_fk, DOWN, buff=0.1)

        explain_ift = stext(
            "IK and FK are inverses — differentiate FK to get IK gradients",
            font_size=18, color=TEXT_COLOR,
        )
        explain_ift.to_edge(DOWN, buff=0.6)

        self.play(FadeIn(ift_title2), FadeIn(explain_ift), run_time=0.5)
        self.play(FadeIn(node_y), FadeIn(node_q), GrowArrow(arrow_ik),
                  FadeIn(ik_label), run_time=1.0)
        self.play(GrowArrow(arrow_fk), FadeIn(fk_label), run_time=0.8)
        self.wait(2.0)
        self.play(FadeOut(explain_ift), run_time=0.3)

        # IFT equation
        ift_eq1 = MathTex(
            r"D\,\text{FK}_A(q) \cdot \frac{\partial q}{\partial y} = I",
            font_size=28, color=ACCENT_4,
        )
        ift_eq1.move_to(DOWN * 0.8)

        ift_eq2 = MathTex(
            r"\Rightarrow\;\; \frac{\partial q}{\partial y} = "
            r"\left[D\,\text{FK}_A(q)\right]^{-1}",
            font_size=28, color=ACCENT_4,
        )
        ift_eq2.next_to(ift_eq1, DOWN, buff=0.3)

        ift_caption = stext(
            "Recover gradients using only FK —\n"
            "no need to differentiate through IK",
            font_size=20, color=ACCENT_3,
        )
        ift_caption.to_edge(DOWN, buff=0.3)

        self.play(Write(ift_eq1), run_time=1.5)
        self.wait(1.5)
        self.play(Write(ift_eq2), run_time=1.5)
        self.play(FadeIn(ift_caption), run_time=0.6)
        self.wait(3.0)

        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)


class RBY1PipelineScene(Scene):
    """RBY1-specific planning pipeline, shown right before the RBY1 experiments."""

    def construct(self):
        self.camera.background_color = BG_COLOR

        pipeline_title = stext("RB-Y1 Planning Pipeline", font_size=32, color=TEXT_COLOR)
        pipeline_title.to_edge(UP, buff=0.5)

        subtitle = stext(
            "Full-body constrained bimanual pick-and-place",
            font_size=20, color=ACCENT_1,
        )
        subtitle.next_to(pipeline_title, DOWN, buff=0.2)

        # "Grasp Selection IK", not "Goal IK": the first stage does not solve
        # for a given goal pose, it *chooses* the grasp -- an optimization IK
        # whose inner loop is the IKFast parameterization. Two lines because
        # the name does not fit one at the shared label size; the boxes are
        # sized for it so all five stay the same shape.
        stages = [
            ("Grasp\nSelection IK", ACCENT_1),
            ("BiRRT", ACCENT_4),
            ("Shortcut", ACCENT_4),
            ("Trajopt", ACCENT_2),
            ("TOPPRA", ACCENT_3),
        ]

        boxes = []
        arrows = []
        box_width = 1.8
        gap = 0.45
        total_w = len(stages) * box_width + (len(stages) - 1) * gap
        start_x = -total_w / 2 + box_width / 2

        for i, (name, color) in enumerate(stages):
            x = start_x + i * (box_width + gap)
            box = RoundedRectangle(
                width=box_width, height=0.95, corner_radius=0.1,
                color=color, stroke_width=2,
                fill_color=color, fill_opacity=0.15,
            )
            box.move_to([x, 0, 0])
            label = stext(name, font_size=18, color=color)
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

        ann_ik = stext("Optimization IK\n+ IKFast", font_size=14, color=ACCENT_1)
        ann_ik.next_to(boxes[0], DOWN, buff=0.4)

        ann_birrt = stext("Sampling-based\npath finding", font_size=14, color=ACCENT_4)
        ann_birrt.next_to(boxes[1], DOWN, buff=0.4)

        ann_trajopt = stext("IFT gradients", font_size=14, color=ACCENT_2)
        ann_trajopt.next_to(boxes[3], DOWN, buff=0.4)

        ann_toppra = stext("Time-optimal\nretiming", font_size=14, color=ACCENT_3)
        ann_toppra.next_to(boxes[4], DOWN, buff=0.4)

        caption = stext(
            "IFT enables gradient-based trajectory optimization\n"
            "through any analytic IK solver",
            font_size=20, color=TEXT_COLOR,
        )
        caption.to_edge(DOWN, buff=0.4)

        self.play(FadeIn(pipeline_title), FadeIn(subtitle), run_time=0.8)
        self.wait(0.5)

        for i, group in enumerate(boxes):
            anims = [FadeIn(group, shift=UP * 0.2)]
            if i > 0:
                anims.append(GrowArrow(arrows[i - 1]))
            self.play(*anims, run_time=0.4)

        self.play(FadeIn(ann_ik), FadeIn(ann_birrt),
                  FadeIn(ann_trajopt), FadeIn(ann_toppra), run_time=0.8)

        self.play(FadeIn(caption), run_time=0.6)
        self.wait(3.0)
        self.play(*[FadeOut(mob) for mob in self.mobjects], run_time=0.8)
