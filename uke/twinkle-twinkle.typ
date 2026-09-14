#import "@preview/conchord:0.4.0": smart-chord, sized-chordlib, chordify


#let uke-chord(..args) = smart-chord(
    tuning: "G1 C1 E1 A1",
    true-bass: false,
    ..args,
)

/*
#box(uke-chord("C"))
#box(uke-chord("G"))
#box(uke-chord("F", n: 1))
#box(uke-chord("G7"))
*/

#place(right, dx: -1em, {
    set align(left)
    // Make all text in chord graph bold
    show text: strong
    // List of used chords there
    sized-chordlib(
        heading-level: 2,
        width: 130pt,
        tuning: "G1 C1 E1 A1",
        true-bass: false,
        switch: (F: 1),
    )
})

#show: chordify

[C] Twinkle, [G] twinkle, [F] little [C] star \
[F] How I [C] wonder [G7] what you [C] are \
[C] Up [G7] above the [C] world so [G7] high \
[C] Like a [G7] diamond [C] in the [G7] sky \
[C] Twinkle, [G] twinkle, [F] little [C] star \
[F] How I [C] wonder [G7] what you [C] are
