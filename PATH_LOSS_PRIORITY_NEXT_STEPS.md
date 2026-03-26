# Path-Loss Priority and Next Steps

This note summarizes why `path_loss` remains the primary target of the project and how the recent review changes the choice of the next experiments.

## Why path loss is still the main priority

- In the local project material, `path_loss` remains the clearest primary objective.
- It is the most mature branch of the project and the easiest one to interpret physically.
- It is also the branch where architectural and loss changes can be compared most cleanly.

In practice, this means:

- spread prediction is still useful and worth improving;
- but path loss remains the branch that should drive the strongest physically motivated development.

## What the recent review says

The latest manual review of the composite panels suggests that the dominant remaining path-loss issue is not simply:

- checkerboard texture,
- local edge detail,
- or missing large-scale context alone.

The stronger reading is:

- the model still misses the **transmitter-centered radial carrier structure** of the propagation field.

In many cases, the model predicts:

- a smooth low-frequency field,
- plus some obstacle-related modulation,

but not the full radial organization that should already exist before obstacle modulation acts on it.

## What follows from that

This means the next path-loss step should not primarily be:

- "make the model bigger",
- "add more attention",
- or "add one more generic regularizer".

The next path-loss step should instead be:

- **supervise the radial propagation structure more explicitly**.

## Why Try 29 is the right immediate follow-up

`Try 29` is the most direct response to the reviewed failure mode.

It keeps the strong clean base from `Try 22` and adds:

- a radial profile loss,
- a radial gradient loss.

That is well aligned with what the data review suggested:

- the model does not only need sharper local details;
- it needs stronger guidance on how the path-loss field should be organized around the transmitter.

## What should come after Try 29 if needed

If `Try 29` only partially improves the problem, the next strongest path-loss direction would likely be:

- a **prior + residual** formulation.

The idea would be:

- build a simple radial or free-space-like prior,
- then train the model to predict the residual over that prior.

This would separate:

- the coarse physical propagation baseline,
- from the environment-specific deviation introduced by obstacles, shadowing, and urban structure.

## Supporting ideas that still make sense

These ideas still look useful, but as secondary additions rather than as the first next path-loss branch:

- bounded output or an explicit out-of-range penalty, because negative denormalized path-loss predictions are physically implausible;
- additional geometry channels such as radial coordinates or other transmitter-centered encodings;
- later, a multitask branch only if it does not obscure the path-loss interpretation.

## Short conclusion

At this point, the most defensible path-loss strategy is:

- keep `path_loss` as the main optimization priority;
- prefer physically targeted supervision over generic extra complexity;
- use `Try 29` as the current best test of the most visible remaining path-loss bottleneck.
