"""Concept anchors: intangible axes written as things a coach says.

Each axis has two poles of short statements in a coach's voice. Embedded
under the same instruction as the pressers, the direction from the mean of
the ``low`` pole to the mean of the ``high`` pole is an axis a presser can be
scored on (the mean projection of its chunks). The statements avoid scores,
statistics, names and injuries on purpose.
"""

from __future__ import annotations

AXES: dict[str, dict[str, tuple[str, ...]]] = {
    "effort": {
        "high": (
            "Our guys played incredibly hard tonight, the effort was there all game.",
            "We competed on every possession and never stopped fighting.",
            "The energy was great from the opening tip, everybody was flying around.",
            "I loved our fight, we got to every loose ball.",
        ),
        "low": (
            "We didn't play hard enough tonight, the effort was not there.",
            "We got outworked, they wanted it more than we did.",
            "Our energy was flat from the start, we were a step slow all night.",
            "We didn't compete, they beat us to every loose ball.",
        ),
    },
    "physicality": {
        "high": (
            "We were the more physical team tonight, we set the tone.",
            "We hit first, we were physical on the ball and on the glass.",
            "Our guys played with force and didn't back down.",
            "We imposed our physicality on them from the start.",
        ),
        "low": (
            "They were more physical than us, they pushed us around.",
            "We got bullied tonight, we didn't match their physicality.",
            "We were soft, we let them be the aggressors.",
            "They hit us first and we never responded.",
        ),
    },
    "ball_movement": {
        "high": (
            "The ball was moving, we played unselfish basketball tonight.",
            "We shared the ball and trusted the pass, everybody touched it.",
            "Our offense had great flow, we made the extra pass all night.",
            "We played together, the ball didn't stick.",
        ),
        "low": (
            "The ball stuck tonight, we played too much one-on-one.",
            "We got stagnant, there was no ball movement.",
            "We settled for isolation and stopped trusting the pass.",
            "Our offense had no flow, everybody was trying to do it alone.",
        ),
    },
    "defensive_focus": {
        "high": (
            "Our defense was connected, we communicated and stuck to the game plan.",
            "We were locked in defensively, our rotations were on time.",
            "We executed the coverages and stayed disciplined all night.",
            "Our attention to detail on defense was excellent.",
        ),
        "low": (
            "We had too many breakdowns on defense, the communication was not there.",
            "We lost our discipline defensively and blew coverages.",
            "Our rotations were late, we were not connected.",
            "We didn't follow the game plan on defense, too many mental mistakes.",
        ),
    },
    "tempo": {
        "high": (
            "We played fast, we got out in transition and pushed the pace.",
            "We ran every chance we got and played with tempo.",
            "We got easy baskets early in the clock by pushing the ball.",
            "We wanted to play in the open floor and we did.",
        ),
        "low": (
            "It was a slow, grind-it-out game played in the half court.",
            "We walked the ball up and played at a slow pace.",
            "We couldn't get out and run, everything was against a set defense.",
            "The game was played possession by possession, late in the clock.",
        ),
    },
    "composure": {
        "high": (
            "We stayed poised when they made their run and kept our composure.",
            "Our guys were calm under pressure and made the right plays late.",
            "We showed maturity, we didn't panic when things went against us.",
            "We executed down the stretch and stayed together.",
        ),
        "low": (
            "We lost our composure when they made their run.",
            "We panicked late and made careless mistakes under pressure.",
            "We let the frustration get to us and fell apart.",
            "We got rattled and stopped executing down the stretch.",
        ),
    },
    "satisfaction": {
        "high": (
            "I'm really proud of this group, that was a great team win.",
            "I'm very happy with how we played tonight.",
            "That's who we want to be, I couldn't ask for more from the guys.",
            "It was a great night for us, a lot to feel good about.",
        ),
        "low": (
            "I'm disappointed, that is not acceptable from us.",
            "That was a bad performance and we have to own it.",
            "I'm frustrated, we are better than what we showed tonight.",
            "It was embarrassing, there is not much to feel good about.",
        ),
    },
    "credit_to_opponent": {
        "high": (
            "Give them credit, they played a great game and made tough shots.",
            "They are a very good team and they deserved it tonight.",
            "Their guys made plays, you have to tip your cap.",
            "They were prepared and executed better than we did, credit to them.",
        ),
        "low": (
            "This was about us, not about what they did.",
            "It had nothing to do with them, we did this to ourselves.",
            "They didn't do anything special, this is on us.",
            "We beat ourselves tonight, it wasn't the other team.",
        ),
    },
}
