inductive Vect (α : Type u) : Nat → Type u where
   | nil : Vect α 0
   | cons : α → Vect α n → Vect α (n + 1)

example : Vect String 3 :=
  .cons "one" (.cons "two" (.cons "three" .nil))

def Vect.reverse_append (lhs : Vect α n) (rhs : Vect α m) : (Vect α (n + m)) :=
  match lhs with
    | .nil         => Nat.zero_add _ ▸ rhs
    | .cons x xs   => Nat.add_assoc .. ▸ reverse_append xs (Nat.add_comm .. ▸ .cons x rhs)

def Vect.reverse (v : Vect α n) : (Vect α n) :=
  reverse_append v nil

def Vect.append (lhs : Vect α n) (rhs : Vect α m) : Vect α (n + m) :=
  reverse_append (reverse lhs) rhs

instance {α : Type} {m n : Nat} : HAppend (Vect α n) (Vect α m) (Vect α (n + m)) where
  hAppend := Vect.append

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
#eval peaks.reverse

def Vect.push : Vect α n -> α -> Vect α (n + 1)
  | .nil, x       => .cons x nil
  | .cons x xs, a => .cons x (.push xs a)

#check Vect.push

-- Naive definitions

def Vect.append_notail : Vect α n → Vect α m → Vect α (n + m)
  | .nil,       bs => Nat.zero_add _ ▸ bs
  | .cons a as, bs => let rest := append_notail as bs
                      Nat.add_assoc .. ▸ Nat.add_comm _ 1 ▸ cons a rest

def Vect.reverse_notail : Vect α n -> Vect α n
  | .nil       => .nil
  | .cons x xs => .push xs.reverse_notail x

-- reverse.go todo acc == (reverse todo) ++ acc

/-
theorem Vect.reverse_append_lemma (a : α) (v : Vect α m) (acc : Vect α n) :
  (cons a v).reverse_notail.append_notail acc =
    Nat.add_assoc .. ▸ append_notail v.reverse_notail (Nat.add_comm .. ▸ cons a acc) := by
  sorry

theorem Vect.reverse_go_lemma (v : Vect α m) : (n : Nat) → (acc : Vect α n) →
  reverse.go v acc = (v.reverse_notail).append_notail acc := by
  induction v with
  | nil =>
    intros
    rfl
  -- unfold append_notail reverse_notail
  | cons a' _ ih =>
    simp [reverse.go]
    intros n acc
    rewrite [reverse_append_lemma]
    simp [*]
-/


/-
theorem vect_reverse_equiv (α : Type) {n : Nat} (v: Vect α n) : (Vect.reverse v) = (Vect.reverse' v) := by
  cases v
  rewrite [Vect.reverse, Vect.reverse', Vect.reverse'.go]
  rfl
  unfold Vect.reverse'
  sorry
-/
