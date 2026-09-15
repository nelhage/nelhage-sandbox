#import "@preview/conchord:0.4.0": smart-chord, sized-chordlib, chordify


#let uke-chord(..args) = smart-chord(
    tuning: "G1 C1 E1 A1",
    true-bass: false,
    ..args,
)

#set text(font: "Helvetica")

#columns(2)[
    #show: chordify

    = Twinkle, Twinkle, Little Star

    \

    [C] Twinkle, [G] twinkle, [F] little [C] star \
    [F] How I [C] wonder [G7] what you [C] are \
    [C] Up [G7] above the [C] world so [G7] high \
    [C] Like a [G7] diamond [C] in the [G7] sky \
    [C] Twinkle, [G] twinkle, [F] little [C] star \
    [F] How I [C] wonder [G7] what you [C] are

    #colbreak()

    #{
        set align(left)
        show text: strong
        sized-chordlib(
            heading-level: 2,
            width: 130pt,
            tuning: "G1 C1 E1 A1",
            true-bass: false,
            switch: (F: 1),
        )
    }
]
