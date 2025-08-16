import Mathlib.Tactic.Common

inductive Vec (α : Type u) : Nat → Type u where
  | nil : Vec α 0
  | cons (x : α) {n : Nat} (xs : Vec α n) : Vec α (n + 1)

variable {α : Type u}

notation "[]" => Vec.nil
infixr:67 " :: " => Vec.cons

/--
Appends `xs` to the front of `ys`.
-/
def Vec.append {m n : Nat} (xs : Vec α m) (ys : Vec α n) : Vec α (n + m):=
  match xs with
  | [] => ys
  | x :: xs' => x :: (xs'.append ys)

/-!
Note: using new notation since the `++` notation isn't good enough to make `[] ++ xs` elaborate`.
It gets stuck trying to figure out how to append `Vec ?m.22 0` and `Vec α m`, and there's sadly no mechanism
that can unify `?m.22` and `α` when `++` is heterogenous.
This `+++` operation does cause the unification to happen.
-/
infixr:65 " +++ " => Vec.append

variable {m m' n n' k k' : Nat} {xs : Vec α m} {ys : Vec α n} {zs : Vec α k}

@[simp]
theorem Vec.nil_append : [] +++ xs = xs := by
  rfl

@[simp]
theorem Vec.cons_append (x : α) :
    (x :: xs) +++ ys = x :: (xs +++ ys) := by
  rfl

/--
Updates the length of a vector using an equality.
Recall that `Vec α m` and `Vec α n` are definitionally equal iff `m` and `n` are definitionally equal.
This function can be used to patch over definitional equality issues.

**Rules:**
* In simp lemmas, casts should appear on the right-hand side, not the left-hand side.
  Exception: simp lemmas specifically about casting.
* Simp lemmas should generally try to move casts "outwards".
-/
def Vec.cast (xs : Vec α m) (h : m = n) : Vec α n :=
  h ▸ xs

@[simp]
theorem Vec.cast_rfl : xs.cast rfl = xs :=
  rfl

@[simp]
theorem Vec.cast_cast {h : m = n} {h' : n = k} : (xs.cast h).cast h' = xs.cast (h.trans h') := by
  subst_vars
  rfl

theorem Vec.cast_eq_symm {h : m = n} : xs.cast h = ys ↔ xs = ys.cast h.symm := by
  subst_vars
  rfl

-- Hint: copy the above proof!
@[simp]
theorem Vec.cons_cast {x : α} {h : m = n} :
    x :: Vec.cast xs h = (x :: xs).cast (by rw [h]) := by
  subst_vars
  rfl

/-!
Think: Why is there no cast theorem for `Vec.nil`?
-/

@[simp]
theorem Vec.append_cast_left {h : m = m'} :
    xs.cast h +++ ys = (xs +++ ys).cast (by rw [h]) := by
  subst_vars
  rfl

@[simp]
theorem Vec.append_cast_right {h : n = n'} :
    xs +++ ys.cast h = (xs +++ ys).cast (by rw [h]) := by
  subst_vars
  rfl

-- Fix this theorem statement by adding a cast to the right-hand side.
@[simp]
theorem Vec.append_nil :
    xs +++ [] = (cast xs (Nat.zero_add m).symm):= by
  induction xs
  rfl
  case cons ih =>
  simp [ih]

-- Same: first fix the theorem statement.
theorem Vec.append_assoc :
    (xs +++ ys) +++ zs = cast (xs +++ (ys +++ zs)) (Nat.add_assoc ..) := by
  induction xs
  simp
  case cons ih =>
  simp [ih]

-- We need two append lemmas since the RHSs have casts.
-- You should be able to use `Vec.cast_eq_symm` to apply `Vec.append_assoc`.
theorem Vec.append_assoc' :
    xs +++ (ys +++ zs) = cast ((xs +++ ys) +++ zs) (Nat.add_assoc ..).symm := by
  rewrite [← cast_eq_symm (h := Nat.add_assoc ..)]
  simp [append_assoc]

theorem Vec.cons_eq_append {x : α} :
    x :: xs = (x :: []) +++ xs := by
  simp [append]

/--
Reverses `xs` and appends it to `ys`.

Tail recursive.
-/
-- We need `n` to be included in the binder list so that they may vary on recursive calls.
-- With `variable` the argument is considered to be fixed.
def Vec.reverseAux {m n : Nat} (xs : Vec α m) (ys : Vec α n) : Vec α (n + m) :=
  match m, xs with
  | 0, [] => ys
  | m' + 1, x :: xs' =>
    let zs : Vec α ((n + 1) + m') := reverseAux xs' (x :: ys)
    zs.cast (by rw [Nat.add_assoc, Nat.add_comm 1])

/--
Reverses `xs`. Tail recursive.
-/
-- Fix the definition by adding a cast.
def Vec.reverse (xs : Vec α m) : Vec α m :=
  (Vec.reverseAux xs []).cast (Nat.zero_add _)

-- It's not necessary to state `Vec.reverseAux_nil` and `Vec.reverseAux_cons` since they're
-- the equation lemmas for `Vec.reverseAux`, but why not.
theorem Vec.reverseAux_nil :
    Vec.reverseAux [] ys = ys :=
  rfl

theorem Vec.reverseAux_cons {x : α} :
    Vec.reverseAux (x :: xs) ys = (reverseAux xs (x :: ys)).cast (by rw [Nat.add_assoc, Nat.add_comm 1]) :=
  rfl

@[simp]
theorem Vec.reverseAux_cast_left (h : m = m') :
    Vec.reverseAux (xs.cast h) ys = (Vec.reverseAux xs ys).cast (by {subst h; rw [Nat.add_comm]} ) := by
  subst_vars
  rfl

@[simp]
theorem Vec.reverseAux_cast_right (h : n = n') :
    Vec.reverseAux xs (ys.cast h) = (Vec.reverseAux xs ys).cast (by {subst h; rw [Nat.add_comm]}) := by
  subst_vars
  rfl

@[simp]
theorem Vec.reverse_nil :
    Vec.reverse [] = ([] : Vec α 0) := by
  unfold reverse reverseAux
  rfl

theorem Vec.reverseAux_eq :
    Vec.reverseAux xs ys = Vec.reverse xs +++ ys := by
  revert n ys
  induction xs
  simp [reverseAux]
  case cons xs' ih =>
  intros
  unfold reverse
  simp [reverseAux_cons]
  simp [ih, append_assoc]

-- Fix and prove
theorem Vec.reverse_cons_append {x : α} :
    (x :: xs).reverse +++ ys = cast (xs.reverse +++ (x :: ys)) (by { rw [Nat.add_assoc, Nat.add_comm 1] }) := by
  unfold reverse
  rewrite [reverseAux_cons]
  simp [reverseAux_eq, append_assoc, cons_append, nil_append]

theorem Vec.reverseAux_append :
    Vec.reverseAux (xs +++ ys) zs = cast (Vec.reverseAux ys (xs.reverse +++ zs)) (by omega) := by
  revert k zs
  induction xs
  intros
  rfl

  case cons ih =>
  intro _ _
  simp [reverseAux_cons]
  rewrite [ih]
  simp [reverse_cons_append]

theorem Vec.reverse_append :
    (xs +++ ys).reverse = (ys.reverse +++ xs.reverse).cast (Nat.add_comm ..) := by
  rewrite (occs := [1]) [reverse]
  rewrite [reverseAux_append]
  simp [reverseAux_eq]


section Count
variable [DecidableEq α]

def Vec.countAux (a : α) {m : Nat} (xs : Vec α m) (acc : Nat) : Nat :=
  match xs with
  | [] => acc
  | x :: xs' => xs'.countAux a (acc + if x = a then 1 else 0)

/-!
Helpful tactic: `split`
It looks for `if` in the goal and splits into multiple goals.

Also: `rename_i` to name the variables produced by `split`.

Also: if you give `simp` a proof or disproof of the hypothesis of an `if`,
it can apply a conditional simp lemma to simplify the `if`.
-/

def Vec.count (a : α) (xs : Vec α m) : Nat :=
  Vec.countAux a xs 0

variable {a : α}

theorem Vec.countAux_eq {acc : Nat} :
    Vec.countAux a xs acc = Vec.count a xs + acc := by
  revert acc
  unfold count
  induction xs <;> unfold countAux
  intro
  rw [Nat.zero_add]

  rename_i ih
  intro
  conv =>
    lhs
    rewrite [ih]
    rhs
    rewrite [Nat.add_comm]
  conv =>
    rhs
    rewrite [ih]
  rw [Nat.add_assoc, Nat.zero_add]

@[simp]
theorem Vec.count_nil :
    ([] : Vec α 0).count a = 0 := by
  unfold count countAux
  rfl

@[simp]
theorem Vec.count_cons_self {x : α} :
    (x :: xs).count x = xs.count x + 1 := by
  unfold count
  rewrite [countAux]
  simp [countAux_eq]

@[simp]
theorem Vec.count_cons_of_eq {x : α} (h : x = a) :
    (x :: xs).count a = xs.count a + 1 := by
  subst_vars
  simp

@[simp]
theorem Vec.count_cons_of_neq {x : α} (h : x ≠ a) :
    (x :: xs).count a = xs.count a := by
  unfold count
  rewrite [countAux]
  simp [h]

@[simp]
theorem Vec.count_cast (h : m = m') :
    Vec.count a (xs.cast h) = Vec.count a xs := by
  subst_vars
  rfl

-- Example of `split`
@[simp]
theorem Vec.count_cons {x : α} :
    (x :: xs).count a = xs.count a + if x = a then 1 else 0 := by
  split <;> simp [*]

theorem Vec.count_append :
    (xs +++ ys).count a = xs.count a + ys.count a := by
  induction xs
  simp
  rename_i ih
  simp [ih]
  conv =>
    rhs
    rewrite [Nat.add_assoc]
    rhs
    rewrite [Nat.add_comm]
  rw [Nat.add_assoc]

theorem Vec.length_reverseAux :
    (Vec.reverseAux xs ys).count a = xs.count a + ys.count a := by
  revert ys n
  induction xs
  intros
  unfold reverseAux
  rw [count_nil, Nat.zero_add]

  rename_i ih
  intro a ys
  rewrite [count_cons]
  conv =>
    rhs
    rewrite [Nat.add_assoc]
    rhs
    rewrite [Nat.add_comm]
    rewrite [← count_cons]
  rewrite [reverseAux_cons, count_cast]
  exact ih

@[simp]
theorem Vec.count_reverse :
    xs.reverse.count a = xs.count a := by
  unfold reverse
  simp [length_reverseAux]

end Count

def Vec.map {β : Type v} (f : α → β) (xs : Vec α m) : Vec β m :=
  sorry

variable {β : Type v} {f : α → β}

/-!
For the following, you may want to develop more theorems.
-/

theorem Vec.map_map {γ : Type _} {g : β → γ} :
    (xs.map f).map g = xs.map (g ∘ f) := by
  sorry

theorem Vec.map_append :
    (xs +++ ys).map f = xs.map f +++ ys.map f := by
  sorry

theorem Vec.reverse_map :
    (xs.map f).reverse = xs.reverse.map f := by
  sorry

def Vec.countP (p : α → Bool) (xs : Vec α m) : Nat :=
  let rec go {m : Nat} (xs : Vec α m) (acc : Nat) : Nat :=
    match xs with
    | [] => acc
    | x :: xs' => go xs' (acc + if p x then 1 else 0)
  go xs 0

-- This `(... · ...)` notation is short for `fun x => (... x ...)`, for making anonymous functions.
theorem Vec.count_eq_countP [DecidableEq α] {a : α} :
    Vec.count a xs = Vec.countP (· = a) xs := by
  sorry

theorem Vec.countP_map (p : β → Bool) :
    (xs.map f).countP p = xs.countP (p ∘ f) := by
  sorry

theorem Vec.count_map [DecidableEq β] {b : β} :
    (xs.map f).count b = xs.countP (f · = b) := by
  sorry

def Vec.filterP (p : α → Bool) {m : Nat} (xs : Vec α m) : Vec α (Vec.countP p xs) :=
  match xs with
  | [] => []
  | x :: xs' =>
    if h : p x then
      (x :: xs'.filterP p).cast sorry
    else
      (xs'.filterP p).cast sorry

variable {p : α → Bool}

theorem Vec.countP_append :
    Vec.countP p (xs +++ ys) = Vec.countP p xs + Vec.countP p ys := by
  sorry

theorem Vec.filterP_append :
    Vec.filterP p (xs +++ ys)
      = (Vec.filterP p xs +++ Vec.filterP p ys).cast (by rw [Vec.countP_append, Nat.add_comm]) := by
  sorry

theorem Vec.filterP_map {p : β → Bool} {f : α → β} :
    Vec.filterP p (Vec.map f xs) = (Vec.map f (Vec.filterP (p ∘ f) xs)).cast sorry := by
  sorry

-- Isn't this `h` hypothesis a little odd?
theorem Vec.filterP_eq_self_iff_countP_eq {h : m = xs.countP p} :
    Vec.filterP p xs = xs.cast h ↔ xs.countP p = m := by
  sorry

-- A way to state the reverse:
theorem Vec.countP_eq_length_iff_filterP_eq_self :
    xs.countP p = m ↔ ∃ (h : m = xs.countP p), Vec.filterP p xs = xs.cast h := by
  sorry

theorem Vec.countP_eq_zero_iff :
    xs.countP p = 0 ↔ ∃ (h : 0 = xs.countP p), Vec.filterP p xs = [].cast h := by
  sorry

theorem Vec.countP_pos_iff_exists :
    0 < Vec.countP p xs
      ↔ ∃ (n n' : Nat) (as : Vec α n) (b : α) (cs : Vec α n') (h : n' + 1 + n = m),
          p b ∧ Vec.countP p as = 0 ∧
          xs = (as +++ (b :: cs)).cast h := by
  sorry

-- For interest, a possible relation to use when equating Vec

def Vec.eq {m n : Nat} (v : Vec α m) (w : Vec α n) : Prop :=
  ∃ h : m = n, v.cast h = w

def Vec.toList {m : Nat} (v : Vec α m) : List α := sorry

def Vec.fromList {m : Nat} (xs : List α) (h : xs.length = m) : Vec α m := sorry
