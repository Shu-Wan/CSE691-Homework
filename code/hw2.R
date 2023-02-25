library(dplyr)
library(ggplot2)
library(purrr)
# Parameters set-up
# n <- 12
# x0 <- 2
# xbar <- 10
# p <- 0.25 # p+ and p-
# beta <- 1.4

n <- 14
x0 <- 3
xbar <- 7
p <- 0.25 # p+ and p-
beta <- 1.4 # return rate

#' Clamp function
#' @description  Constrain a number between `x0` and `xbar`
#'
#' @param x number
#' @return constrained number
clamp <- function(x) {
  return(max(min(x, xbar), 0))
}

#' Optimal reward-to-go function
#' @description Exact DP
#'
#' @param x starting state
#' @param t starting time
#' @return optimal reward-to-go values
reward <- function(x, t) {
  stopifnot(x == clamp(x))
  stopifnot(t <= n)

  if (t == n) {
    return(x)
  }

  if (x == xbar) {
    return(x)
  }

  e <- p * reward(clamp(x + 1), t + 1) +
    p * reward(clamp(x - 1), t + 1) +
    (1 - 2 * p) * reward(x, t + 1)

  return(max(x, e))
}

#' Plotting function
plot_value <- function(space, title = "") {
  space %>%
    ggplot(aes(time, state)) +
    geom_tile(aes(fill = reachable), color = "black", 
    linewidth = .5, show.legend = F) +
    geom_text(aes(label = round(value, 3)), color = "white", 
    fontface = "bold") +
    theme_minimal() +
    scale_x_continuous(breaks = 0:n) +
    scale_y_continuous(breaks = 0:xbar) +
    scale_fill_manual(values = c("#FFFFFF", "#808080")) +
    labs(title = title, x = "Time", y = "State (Price)") +
    theme(panel.grid = element_blank(), 
    plot.title = element_text(face = "bold", hjust = .5))
}
plot_policy <- function(space, title = "") {
  space %>%
    ggplot(aes(time, state)) +
    geom_tile(aes(fill = reachable), color = "black", 
    linewidth = .1, show.legend = F) +
    geom_text(aes(label = policy, color = policy), 
    fontface = "bold", show.legend = F) +
    theme_minimal() +
    scale_x_continuous(breaks = 0:n) +
    scale_y_continuous(breaks = 0:xbar) +
    scale_fill_manual(values = c("#FFFFFF", "#808080")) +
    scale_color_manual(values = c("#FFFFFF", "#FFC627")) +
    labs(title = title, x = "Time", y = "State (Price)") +
    theme(panel.grid = element_blank(), 
    plot.title = element_text(face = "bold", hjust = .5))
}

#' Base heuristic algorithm
#' @description Base heuristic algorithm depend on the referencing state
#' @param x starting state
#' @param t starting time
#' @param xk referencing state
#' @return expected reward-to-go J_k^k
base_heuristic <- function(x, t, xk = x) {
  # stop if x and xk are out of boundary
  stopifnot(x == clamp(x), xk == clamp(xk))
  stopifnot(t <= n)

  if (t == n) {
    return(x)
  }

  # sell condition: current price exceeds the selling threshold
  if (x > beta * xk) {
    return(x)
  }
  # sell condition: reach xbar
  if (x == xbar) {
    return(x)
  }

  e <- p * base_heuristic(clamp(x + 1), t + 1, xk) +
    p * base_heuristic(clamp(x - 1), t + 1, xk) +
    (1 - 2 * p) * base_heuristic(x, t + 1, xk)

  return(max(x, e))
}

#' One-step rollout minimization algorithm
#' @param x starting state
#' @param t starting time
#' @return expected reward-to-go values
one_step <- function(x, t) {
  stopifnot(x == clamp(x)) # stop if x is out of boundary

  if (t == n) {
    return(x)
  }

  # sell condition: reach xbar
  if (x == xbar) {
    return(x)
  }

  e <- p * base_heuristic(clamp(x + 1), t + 1) +
    p * base_heuristic(clamp(x - 1), t + 1) +
    (1 - 2 * p) * base_heuristic(x, t + 1)
  return(max(x, e))
}

# monte carlo simulation with base heuristic
#' @param x
#' @param t
#' @param xk the referencing price
#' @param sim
#' @return expected reward-to-go
monte_carlo_reward <- function(x, t, xk = x, sim) {
  stopifnot(x == clamp(x), xk == clamp(xk))
  stopifnot(t <= n)

  one_sim <- function(x, t) {
    while (t < n) {
      # terminate if reaching the cap
      if (x == xbar) {
        return(x)
        break
      }
      # terminate by base heuristic
      if (x > beta * xk) {
        return(x)
        break
      }
      rng <- runif(1, 0, 1)
      if (rng < p) {
        x <- x + 1
      } else if (rng > (1 - p)) {
        x <- clamp(x - 1)
      }
      t <- t + 1
    }
    return(x)
  }

  mean(replicate(sim, one_sim(x, t)))
}

#' One-step rollout using Monte Carlo approximation
one_step_mc <- function(x, t, sim) {
  stopifnot(x == clamp(x)) # stop if x is out of boundary

  if (t == n) {
    return(x)
  }

  e <- p * monte_carlo_reward(clamp(x + 1), t + 1, sim = sim) +
    p * monte_carlo_reward(clamp(x - 1), t + 1, sim = sim) +
    (1 - 2 * p) * monte_carlo_reward(x, t + 1, sim = sim)
  return(max(x, e))
}

#' Two-step lookahead algorithm
#' @param x starting state
#' @param t starting time
#' @return expected reward-to-go values
two_step <- function(x, t) {
  stopifnot(x == clamp(x)) # stop if x and xk are out of boundary

  if (t == n) {
    return(x)
  }

  if (t == n - 1) {
    return(one_step(x, t)) # replace with one-step lookahead
  }

  e <- p * p * base_heuristic(clamp(x + 2), t + 2) +
    2 * p * (1 - 2 * p) * base_heuristic(clamp(x + 1), t + 2) +
    (p^2 + (1 - 2 * p)^2) * base_heuristic(x, t + 2) +
    2 * p * (1 - 2 * p) * base_heuristic(clamp(x - 1), t + 2) +
    p * p * base_heuristic(clamp(x - 2), t + 2)
  return(max(x, e))
}

# create state space
space <- expand_grid(time = 0:n, state = 0:xbar) %>%
  mutate(value = NA, policy = NA, reachable = NA) %>%
  mutate(reachable = (abs(state - x0) <= time))

# part(a): exact DP
spacea <- space %>%
  mutate(value = ifelse(reachable, map2_dbl(state, time, reward), NA)) %>%
  mutate(policy = ifelse(ifelse(reachable, state == value, NA), "Sell", "Keep"))

plot_value(spacea, "Expected Reward-to-go (Exact DP)")
ggsave("parta1.png")
plot_policy(spacea, "Policy (Exact DP)")
ggsave("parta2.png")

# part(b): base heuristic
spaceb <- space %>%
  mutate(value = ifelse(reachable, 
  map2_dbl(state, time, base_heuristic, xk = x0), NA)) %>%
  mutate(policy = ifelse(ifelse(reachable, state == value, NA), "Sell", "Keep"))

plot_value(spaceb, "Expected Reward-to-go (Base heuristic)")
ggsave("partb1.png")
plot_policy(spaceb, "Policy (Base heuristic)")
ggsave("partb2.png")

# part(c): one-step rollout and monte carlo approximation
spacec <- space %>%
  mutate(value = ifelse(reachable, map2_dbl(state, time, one_step), NA)) %>%
  mutate(policy = ifelse(ifelse(reachable, state == value, NA), "Sell", "Keep"))

plot_value(spacec, "Expected Reward-to-go (One-step rollout)")
ggsave("partc1.png")
plot_policy(spacec, "Policy (One-step rollout)")
ggsave("partc2.png")

one_step_mc(x0, 0, sim = 500L)
one_step_mc(x0, 0, sim = 1e3)
one_step_mc(x0, 0, sim = 1e5)

# part(d): two-step lookahead
spaced <- space %>%
  mutate(value = ifelse(reachable, map2_dbl(state, time, two_step), NA)) %>%
  mutate(policy = ifelse(ifelse(reachable, state == value, NA), "Sell", "Keep"))

plot_value(spaced, "Expected Reward-to-go (Two-step lookahead)")
ggsave("partd2.png")
plot_policy(spaced, "Policy (Two-step lookahead)")
ggsave("partd2.png")
