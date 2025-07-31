def Day  := Nat    -- Days since some epoch
def Hour := Fin 24 -- Fin 24 represents a number in [0, 24)
def USD  := Float  -- Don't lecture me, it's a toy example

structure AirportStats where
  name: String
  flights: Nat
  revenue: USD

-- Straightforward interpretation: A table with two dimensions,
-- returning a structure
def AirportTable := Day → Hour → AirportStats

-- A type representing the fields of AirportStats. We will use this to
-- lift the struct dimension into a function argument like the others
--
-- See: https://lean-lang.org/functional_programming_in_lean/Programming-with-Dependent-Types/The-Universe-Design-Pattern/#Functional-Programming-in-Lean--Programming-with-Dependent-Types--The-Universe-Design-Pattern
inductive AirportStats.Field where
  | name
  | flights
  | revenue

-- The dependent-typing magic starts: Given a field,
-- compute its type
def AirportStats.Field.asType : AirportStats.Field → Type
  | .name => String
  | .flights => Nat
  | .revenue => USD

-- Now we can use that definition to lift the struct axis into a "normal" dimension
def AirportTable.Lifted := Day → Hour → (f : AirportStats.Field) → f.asType

-- And the implementation to do the lifting.
-- First, a helper to lift a single stats object:
def AirportStats.asTable (a : AirportStats) : (f : AirportStats.Field) → f.asType
  | .name => a.name
  | .flights => a.flights
  | .revenue => a.revenue

-- And now we can use that to lift the entire table. We could also do
-- fancier metaprogramming to make this more generic, of course.
def AirportTable.liftStruct (tabl : AirportTable) : AirportTable.Lifted :=
  fun day hour => (tabl day hour).asTable

-- We can also move dimensions around. Here we move the "field" first,
-- making it "SoA"-style
def AirportTable.toSoA (tab : AirportTable) : (f : AirportStats.Field) → (d : Day) → (h : Hour) → f.asType :=
  fun f d h => (tab.liftStruct) d h f
