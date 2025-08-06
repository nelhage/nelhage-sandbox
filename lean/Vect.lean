inductive Vect (α : Type u) : Nat → Type u where
   | nil : Vect α 0
   | cons : α → Vect α n → Vect α (n + 1)

example : Vect String 3 :=
  .cons "one" (.cons "two" (.cons "three" .nil))

namespace Vect

def cast {m n : Nat} (xs : Vect α m) (he : m = n) : (Vect α n) :=
  _root_.cast (congrArg (Vect α) he) xs

@[simp]
theorem cast_cast {m n p : Nat} (v : Vect α m) (he₁ : m = n) (he₂ : n = p) :
  (v.cast he₁).cast he₂ = v.cast (Eq.trans he₁ he₂) := by
  simp [cast]

@[simp]
theorem cast_eq {n : Nat} {xs: Vect α n} : xs.cast rfl = xs := by
  rfl

def reverse_append (lhs : Vect α n) (rhs : Vect α m) : (Vect α (n + m)) :=
  match lhs with
    | .nil         => cast rhs (by omega)
    | .cons x xs   => cast (reverse_append xs (.cons x rhs)) (by omega)

def reverse (v : Vect α n) : (Vect α n) :=
  reverse_append v nil

def append (lhs : Vect α n) (rhs : Vect α m) : Vect α (n + m) :=
  reverse_append (reverse lhs) rhs

def head : (v : Vect α (Nat.succ n)) → α
  | cons a _ => a

def head' : (v : Vect α n) → (hn : n = Nat.succ m) → α
  | cons a _, _ => a

def tail : {n: Nat} → (v : Vect α (Nat.succ n)) → α
  | 0,     cons a nil  => a
  | _ + 1, cons _ rest => rest.tail

instance {α : Type} {m n : Nat} : HAppend (Vect α n) (Vect α m) (Vect α (n + m)) where
  hAppend := append

def zip : Vect α n → Vect β n → Vect (α × β) n
  | .nil, .nil => .nil
  | .cons x xs, .cons y ys => .cons (x, y) (zip xs ys)

def map : (α → β) → Vect α n → Vect β n
  | _, .nil => .nil
  | f, .cons x xs => .cons (f x) (.map f xs)

def zipWith : (α → β → γ) → Vect α n → Vect β n → Vect γ n
  | f, .nil, .nil => .nil
  | f, .cons x xs, .cons y ys => .cons (f x y) (.zipWith f xs ys)


-- def peaks : Vect String 3 := .cons "Rainier" (.cons "Hood" (.cons "Denali" .nil))
-- #eval peaks.tail
-- #eval peaks.map fun x => "Mount " ++ x
-- #eval peaks.reverse

def push : Vect α n -> α -> Vect α (n + 1)
  | .nil, x       => .cons x nil
  | .cons x xs, a => .cons x (.push xs a)

-- #check Vect.push

-- Naive definitions

def append_notail : Vect α n → Vect α m → Vect α (n + m)
  | .nil,       bs => Nat.zero_add _ ▸ bs
  | .cons a as, bs => let rest := append_notail as bs
                      Nat.add_assoc .. ▸ Nat.add_comm _ 1 ▸ cons a rest

theorem reverse_nil {α : Type} :
  reverse (nil : Vect α 0) = nil := by
  rfl

theorem zero_is_nil.{u} {α : Type u} (v : Vect α 0) : v = nil := by
  cases v
  rfl

protected theorem move_add1 (n m : Nat) : (n + 1) + m = (n + m) + 1 := by
  omega

@[simp]
theorem cast_head {m n : Nat} (h : (Nat.succ m) = (Nat.succ n)) (v : Vect α (Nat.succ m)) :
  head (v.cast h : Vect α (Nat.succ n)) = head v := by
  cases h
  simp [head]

@[simp]
theorem head_cons (a : α) (v : Vect α n) :
  head (cons a v) = a := by
  simp [head]

theorem reverse_append_head (v : Vect α (Nat.succ n)) : {m : Nat} → (b : Vect α m) → (c : Vect α m) →
  ((v.reverse_append b).cast (by omega) : Vect α (n + m + 1)).head =
  ((v.reverse_append c).cast (by omega) : Vect α (n + m + 1)).head := by
  induction n
  have (.cons a nil) := v
  simp [reverse_append]
  case succ ih =>
    have (.cons v vs) := v
    intros m' b c
    simp only [reverse_append, cast_cast]
    have ihv := (ih vs (m := m'.succ) (cons v b) (cons v c))
    simp
    simp at ihv
    exact ihv

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
