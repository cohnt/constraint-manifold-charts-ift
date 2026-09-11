"""Title, authorship and affiliation text shared by both videos' cards.

Named `video_credits` rather than `credits` on purpose: `credits` is already a
builtin (site._Printer), so a module by that name fails only at the first
attribute access, with a confusing AttributeError on a `_Printer`, instead of
raising ImportError where the mistake is.


One place for anything that names the paper or its authors, so the two videos
cannot drift apart and so anonymisation cannot be applied to one card and
forgotten on another.

## Anonymity

The RA-L submission is double-blind, so the results video **defaults to
anonymous** and has to be asked for names explicitly (`--named`). The defaults
are deliberately asymmetric: shipping names to a double-blind review is a far
worse failure than shipping an anonymous copy of a promotional video, so the
safe state is the one you get by forgetting the flag.

  * `make_cards.py`    (RA-L supplementary) -- anonymous unless `--named`
  * `make_cards_v2.py` (overview / LinkedIn) -- named unless `--anonymous`
"""

TITLE_LINES = [
    "Planning along Differentiable Charts of",
    "Constraint Manifolds with General-Purpose IK Solvers",
]

# Wrapped differently on the overview card, which uses a narrower measure.
TITLE_LINES_NARROW = [
    "Planning along Differentiable Charts of",
    "Constraint Manifolds with",
    "General-Purpose IK Solvers",
]

# (name, is_co_first). Thomas Cohn and Seiji Shaw contributed equally and are
# both marked; the asterisk is explained by EQUAL_CONTRIBUTION_NOTE, which must
# be shown wherever the asterisks are.
AUTHORS = [
    ("Thomas Cohn", True),
    ("Seiji Shaw", True),
    ("Harel Biggie", False),
    ("Travis Manderson", False),
    ("Nicholas Roy", False),
    ("Russ Tedrake", False),
]

EQUAL_CONTRIBUTION_NOTE = "* Denotes equal contribution"

AFFILIATION = "Massachusetts Institute of Technology  —  CSAIL"
AFFILIATION_SHORT = "MIT CSAIL"
VENUE = "IEEE Robotics and Automation Letters"

ANONYMOUS_AUTHORS = "Anonymous Authors"
ANONYMOUS_AFFILIATION = "Paper under double-blind review"


def author_line(separator="    "):
    """The author list with co-first asterisks, e.g. 'Thomas Cohn*    ...'."""
    return separator.join(
        name + ("*" if co_first else "") for name, co_first in AUTHORS
    )


def credit_lines(anonymous, affiliation=AFFILIATION, venue=VENUE,
                 separator="    "):
    """The block of lines under the title, top to bottom.

    Returns a list of (text, role) pairs. `role` is one of "authors", "note",
    "affiliation" or "venue", so each card can pick its own font and colour
    without duplicating the decision about *what* appears.

    When anonymous, the author names, the equal-contribution note (which is
    itself about specific authors) and the affiliation are all replaced -- not
    merely the names. A card showing "* Denotes equal contribution" under
    "Anonymous Authors" would leak that there are exactly two co-first authors.
    """
    if anonymous:
        lines = [(ANONYMOUS_AUTHORS, "authors"),
                 (ANONYMOUS_AFFILIATION, "affiliation")]
    else:
        lines = [(author_line(separator), "authors"),
                 (EQUAL_CONTRIBUTION_NOTE, "note")]
        if affiliation:
            lines.append((affiliation, "affiliation"))
    if venue:
        lines.append((venue, "venue"))
    return lines
