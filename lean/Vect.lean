inductive Vect (α : Type u) : Nat → Type u where
   | nil : Vect α 0
   | cons : α → Vect α n → Vect α (n + 1)

example : Vect String 3 :=
  .cons "one" (.cons "two" (.cons "three" .nil))

def Vect.zip : Vect α n → Vect β n → Vect (α × β) n
  | .nil, .nil => .nil
  | .cons x xs, .cons y ys => .cons (x, y) (zip xs ys)

def Vect.map : (α → β) → Vect α n → Vect β n
  | _, .nil => .nil
  | f, .cons x xs => .cons (f x) (.map f xs)

def Vect.zipWith : (α → β → γ) → Vect α n → Vect β n → Vect γ n
  | f, .nil, .nil => .nil
  | f, .cons x xs, .cons y ys => .cons (f x y) (.zipWith f xs ys)

def peaks : Vect String 3 := .cons "Rainier" (.cons "Hood" (.cons "Denali" .nil))

#eval peaks.map fun x => "Mount " ++ x

def Vect.push : Vect α n -> α -> Vect α (n + 1)
  | .nil, x => .cons x nil
  | .cons x xs, a => .cons x (.push xs a)

#check Vect.push

def Vect.reverse : Vect α n -> Vect α n
  | .nil => .nil
  | .cons x xs => .push xs.reverse x

#eval peaks.reverse

#check Nat.sub_self

def Vect.reverse' (v : Vect α n) : Vect α n :=
  let rec go {m n : Nat} (todo : Vect α n) (acc : Vect α m) : (Vect α (n + m)) :=
    match todo with
      | .nil         => Nat.zero_add _ ▸ acc
      | .cons x xs   => Nat.add_assoc .. ▸ go xs (Nat.add_comm .. ▸ .cons x acc)
  go v nil
