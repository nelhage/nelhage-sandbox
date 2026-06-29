"use strict";
// JS port of summands.py: a base class with three subclasses and polysum,
// which loops calling s.add(total) -- a polymorphic call site when the list
// holds a mix of types.

class Summand {
  constructor(value) {
    this.value = value;
  }
  add(total) {
    throw new Error("not implemented");
  }
}

class AddInt extends Summand {
  add(total) {
    return total + this.value;
  }
}

class SquareInt extends Summand {
  add(total) {
    return total + this.value * this.value;
  }
}

class Identity extends Summand {
  add(total) {
    return total;
  }
}

const KINDS = [AddInt, SquareInt, Identity];

function polysum(summands) {
  let total = 0;
  for (const s of summands) {
    total = s.add(total);
  }
  return total;
}

module.exports = { Summand, AddInt, SquareInt, Identity, KINDS, polysum };
