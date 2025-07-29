inductive Vect (α : Type u) : Nat → Type u where
   | nil : Vect α 0
   | cons : α → Vect α n → Vect α (n + 1)

example : Vect String 3 :=
  .cons "one" (.cons "two" (.cons "three" .nil))
