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
theorem cast_rfl {n : Nat} {xs: Vect α n} : xs.cast rfl = xs := by
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

-- #check push

-- Naive definitions

def append_notail : Vect α n → Vect α m → Vect α (m + n)
  | .nil,       bs => bs
  | .cons a as, bs => cons a (append_notail as bs)

@[simp]
theorem reverse_nil {α : Type} :
  reverse (nil : Vect α 0) = nil := by
  rfl

@[simp]
theorem zero_is_nil.{u} {α : Type u} (v : Vect α 0) : v = nil := by
  cases v
  rfl

@[simp]
theorem cast_head {m n : Nat} (h : (Nat.succ m) = (Nat.succ n)) (v : Vect α (Nat.succ m)) :
  head (v.cast h : Vect α (Nat.succ n)) = head v := by
  cases h
  simp [head]

@[simp]
theorem head_cons (a : α) (v : Vect α n) :
  head (cons a v) = a := by
  simp [head]

theorem reverse_append_head (v : Vect α (Nat.succ n)) :
  {o p : Nat} → (b : Vect α o) → (c : Vect α p) →
  let vb := (cast (v.reverse_append b) (by omega) : Vect α (n + o + 1));
  let vc := (cast (v.reverse_append c) (by omega) : Vect α (n + p + 1));
    vb.head = vc.head := by
  induction n
  have (.cons a nil) := v
  simp [reverse_append]
  case succ ih =>
    have (.cons v vs) := v
    intros _ _ b c
    simp [reverse_append]
    have ihv := (ih vs (cons v b) (cons v c))
    simp at ihv
    exact ihv

@[simp]
theorem cons_reverse_head (a : α) (v : Vect α (n + 1)) :
  (cons a v).reverse.head = v.reverse.head := by
  simp [reverse, reverse_append]
  have rah := reverse_append_head v (cons a nil) nil
  simp at rah
  assumption

@[simp]
theorem head_reverse {α : Type} {n : Nat} :
  (v : Vect α (n + 1)) → (head (reverse v)) = (tail v) := by
  intro v
  induction n with
  | zero =>
    cases v
    simp [zero_is_nil] at *
    rfl
  | succ n' ih =>
    have (.cons a xs) : Vect α (n' + 1 + 1) := v
    rewrite [tail, cons_reverse_head]
    apply ih

@[simp]
theorem reverse_append_cons {n m : Nat} (a : α) (v : Vect α n) : (rhs : Vect α m) →
  (cons a v).reverse_append rhs =
    (cast (v.reverse_append (cons a rhs)) (by omega)) := by
  simp [reverse_append]

@[simp]
theorem cast_reverse_append {n m o: Nat} (v : Vect α n) (w : Vect α o) (h : n = m) :
  (v.cast h).reverse_append w = (v.reverse_append w).cast (by omega) := by
  subst_vars
  simp [reverse_append]

@[simp]
theorem cons_cast {n m : Nat} (a : α) (v : Vect α n) (h : n = m) :
  (cons a (cast v h)) = (cons a v).cast (by omega) := by
  subst_vars
  simp

theorem cast_eq_symm {n m : Nat} (lhs : Vect α n) (rhs : Vect α m) (h : n = m) :
  lhs.cast h = rhs ↔ lhs = rhs.cast h.symm := by
  subst_vars
  rfl

theorem cons_push {n : Nat} : (a b : α) → (v : Vect α n) →
  push (cons a v) b  = cons a (push v b) := by
  induction n
  simp [push]
  case succ ih =>
  intro a b v
  have (.cons v vs) := v
  rewrite [push]
  congr

theorem push_reverse_append {n m : Nat} (a : α) (xs : Vect α n) (ys : Vect α m) :
  (xs.push a).reverse_append ys = (cons a (xs.reverse_append ys)).cast (by omega) := by
  revert m
  induction n
  cases xs
  simp [reverse_append, push]
  case succ ih =>
  have (.cons x xs') := xs
  intro m ys
  unfold push
  unfold reverse_append
  have ihv := ih xs' (cons x ys)
  simp [cast_eq_symm]
  assumption

theorem reverse_append_push_reverse_append {n : Nat} (a : α)
  (xs : Vect α n) : {m o : Nat} → (ys : Vect α m) → (zs: Vect α o) →
  (xs.reverse_append (push ys a)).reverse_append zs =
    (cast (cons a ((xs.reverse_append ys).reverse_append zs)) (by omega)) := by
  induction n
  cases xs
  intros
  simp [reverse_append]
  simp [cast_eq_symm]
  rewrite [push_reverse_append]
  simp

  case succ ih =>
  intros m o ys zs
  have (.cons x xs') := xs
  simp [reverse_append]
  rewrite [← cons_push]
  have ihv := ih xs' (cons x ys) zs
  rewrite [cast_eq_symm]
  simp
  exact ihv


@[simp]
theorem reverse_append_nil {n : Nat} (xs : Vect α n) :
  (xs.reverse_append nil).reverse_append nil = xs := by
  induction n
  simp [reverse_append]
  case succ ih =>
  have (.cons x xs') := xs
  simp [reverse_append]
  rewrite [← push]
  rewrite [reverse_append_push_reverse_append]
  rewrite [cast_eq_symm]
  rewrite [ih]
  simp


theorem reverse_append_is_reverse_append {n m : Nat} (xs : Vect α n) (ys : Vect α m) :
  xs.reverse_append ys = (xs.reverse) ++ ys := by
  simp [(· ++ ·)]
  simp [reverse, append]

theorem reverse_reverse {n : Nat} (v : Vect α n) :
  v.reverse.reverse = v := by
  unfold reverse
  simp


theorem append_equivalent : {n m : Nat} → (lhs : Vect α n) → (rhs : Vect α m) →
  lhs.append rhs = cast (lhs.append_notail rhs) (Nat.add_comm ..) := by
  intro n
  induction n
  intros _ lhs
  cases lhs
  simp [append, reverse_append, append_notail]

  case succ ih =>
  intro m lhs rhs
  have (.cons l ls) := lhs
  unfold append reverse
  simp [reverse_append_cons]
  rewrite [← push]
  rewrite [reverse_append_push_reverse_append]
  rewrite [← reverse, ← append, cast_eq_symm]
  unfold append_notail
  rewrite [ih ls rhs]
  simp

-- Theorems we'd like to prove:
-- xs.append ys = xs.append_notail ys
-- (reverse (reverse v)) = v


-- (x.reverse_append y).reverse_append z
--  = ((x.reverse) ++ y).reverse ++ z
--  = (y.reverse ++ x) ++ z
--  = y.reverse_append (x ++ z)
