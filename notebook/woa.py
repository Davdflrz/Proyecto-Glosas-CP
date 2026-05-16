"""
Whale Optimization Algorithm — Implementación propia para tesis de maestría.

Basada en:
    Mirjalili, S., & Lewis, A. (2016). The Whale Optimization Algorithm.
    Advances in Engineering Software, 95, 51-67.
    DOI: 10.1016/j.advengsoft.2016.01.008

Esta implementación NO usa librerías de metaheurísticos (mealpy/niapy/etc.):
todo el algoritmo está escrito desde cero en NumPy puro, siguiendo las
ecuaciones 2.1-2.7 del paper original.

------------------------------------------------------------
Modificaciones originales aplicadas al contexto de glosas:
------------------------------------------------------------
1) Latin Hypercube Sampling (LHS) para inicialización
   - Reemplaza el muestreo uniforme del paper original.
   - Garantiza cobertura estratificada del espacio de búsqueda.
   - Referencia: McKay, Beckman & Conover (1979).

2) Schedule cosine para el coeficiente 'a'
   - Reemplaza la disminución lineal del paper.
   - Transición más suave entre exploración y explotación.
   - Inspirado en Loshchilov & Hutter (2017) — cosine annealing.

3) Elitismo de top-2 ballenas
   - Preserva las dos mejores soluciones entre épocas.
   - Evita pérdida de buenos óptimos por la naturaleza estocástica
     del muestreo. Referencia: De Jong (1975), Goldberg (1989).

4) Greedy selection
   - Solo acepta una nueva posición si su fitness mejora la actual.
   - Acelera convergencia frente al esquema "siempre actualizar"
     del paper original.

------------------------------------------------------------
Parámetros calibrados para este problema específico:
------------------------------------------------------------
- b = 1.5  (vs default 1.0)
  Justificación: superficie de fitness ruidosa por CV de 3 pliegues
  → espirales más amplias para escapar de óptimos locales falsos.

- pop_size = 18  (~2.5 × dim, siendo dim=7)
  Justificación: 7 hiperparámetros XGBoost requieren mayor diversidad
  que el default de 10-15 ballenas.

- epoch = 25  (vs 20-30 típicos)
  Justificación: compensa el ruido del CV interno (3 folds estocásticos)
  permitiendo mayor convergencia estadística.

- n_elites = 2  (~11% de la población)
  Justificación: ratio elitista estándar entre 5-15% en literatura GA.
"""

import numpy as np


class WhaleOptimizationAlgorithm:
    """
    Implementación propia del Whale Optimization Algorithm con 4 mejoras
    adaptadas al problema de optimización de hiperparámetros XGBoost para
    predicción de glosas médicas.
    """

    def __init__(self, obj_func, bounds, epoch=25, pop_size=18,
                 b=1.5, n_elites=2, seed=42):
        """
        Parámetros
        ----------
        obj_func : callable
            Función a minimizar. Recibe un array NumPy de tamaño dim
            y retorna un escalar.
        bounds : list of (lb, ub)
            Lista de tuplas con límites inferior y superior por dimensión.
        epoch : int
            Número de iteraciones del algoritmo.
        pop_size : int
            Tamaño de la población de ballenas.
        b : float
            Constante del logaritmo espiral (ecuación 2.6 del paper).
        n_elites : int
            Cantidad de mejores ballenas a preservar entre épocas (elitismo).
        seed : int
            Semilla para reproducibilidad.
        """
        self.obj_func = obj_func
        self.bounds = np.asarray(bounds, dtype=float)
        self.dim = len(bounds)
        self.epoch = epoch
        self.pop_size = pop_size
        self.b = b
        self.n_elites = n_elites
        self.rng = np.random.default_rng(seed)

        # Estado del algoritmo
        self.best_solution = None
        self.best_fitness = np.inf
        self.history = []  # mejor fitness por época

    # ------------------------------------------------------------------
    # Inicialización con Latin Hypercube Sampling (modificación 1)
    # ------------------------------------------------------------------
    def _latin_hypercube_init(self):
        """
        Latin Hypercube Sampling: divide cada dimensión en pop_size
        estratos y muestrea uno por estrato. Garantiza cobertura
        balanceada del espacio frente al muestreo uniforme tradicional.
        """
        lb = self.bounds[:, 0]
        ub = self.bounds[:, 1]
        result = np.zeros((self.pop_size, self.dim))
        for d in range(self.dim):
            # Crear estratos uniformes y aplicar permutación aleatoria
            strata = np.linspace(0, 1, self.pop_size + 1)
            samples = self.rng.uniform(strata[:-1], strata[1:])
            self.rng.shuffle(samples)
            result[:, d] = lb[d] + samples * (ub[d] - lb[d])
        return result

    # ------------------------------------------------------------------
    # Schedule cosine del coeficiente 'a' (modificación 2)
    # ------------------------------------------------------------------
    def _cosine_a(self, t):
        """
        Decrece 'a' de 2 a 0 con curva cosenoidal en lugar de lineal.
        Provee mayor tiempo en zona de exploración antes de explotar.

        Original (paper):    a = 2 - 2·t/T
        Modificado (cosine): a = 1 + cos(π·t/T)
        """
        return 1.0 + np.cos(np.pi * t / self.epoch)

    def _clip(self, x):
        """Reflejar al borde del espacio si la solución se sale."""
        return np.clip(x, self.bounds[:, 0], self.bounds[:, 1])

    # ------------------------------------------------------------------
    # Bucle principal del algoritmo
    # ------------------------------------------------------------------
    def optimize(self, verbose=True):
        """
        Ejecuta el ciclo WOA con las 4 modificaciones aplicadas.
        Retorna la mejor solución encontrada, su fitness y el historial.
        """
        # ===== INICIALIZACIÓN (modificación 1: LHS) =====
        population = self._latin_hypercube_init()
        fitness = np.array([self.obj_func(x) for x in population])

        # Mejor agente global
        best_idx = np.argmin(fitness)
        self.best_solution = population[best_idx].copy()
        self.best_fitness = float(fitness[best_idx])
        self.history.append(self.best_fitness)

        if verbose:
            print(f"  Inicial (LHS): mejor fitness = {self.best_fitness:.6f}")

        # ===== CICLO PRINCIPAL =====
        for t in range(self.epoch):
            a = self._cosine_a(t)  # modificación 2: cosine schedule

            for i in range(self.pop_size):
                # Coeficientes aleatorios A y C (ecuaciones 2.3 y 2.4)
                r1 = self.rng.random(self.dim)
                r2 = self.rng.random(self.dim)
                A = 2.0 * a * r1 - a   # eq 2.3
                C = 2.0 * r2           # eq 2.4

                # p decide entre mecanismos (paper: p ~ U[0,1])
                p = self.rng.random()
                # l parametriza la espiral logarítmica (eq 2.6)
                l = self.rng.uniform(-1.0, 1.0, self.dim)

                if p < 0.5:
                    if np.linalg.norm(A) < 1.0:
                        # ── Cerco a la presa (exploitation) — ec. 2.1, 2.2 ──
                        D = np.abs(C * self.best_solution - population[i])
                        new_x = self.best_solution - A * D
                    else:
                        # ── Búsqueda de presa (exploration) — ec. 2.7 ──
                        rand_idx = self.rng.integers(0, self.pop_size)
                        x_rand = population[rand_idx]
                        D = np.abs(C * x_rand - population[i])
                        new_x = x_rand - A * D
                else:
                    # ── Ataque en burbujas espirales — ec. 2.5, 2.6 ──
                    D_prime = np.abs(self.best_solution - population[i])
                    new_x = (D_prime * np.exp(self.b * l) *
                             np.cos(2.0 * np.pi * l) + self.best_solution)

                new_x = self._clip(new_x)

                # ===== GREEDY SELECTION (modificación 4) =====
                new_fit = self.obj_func(new_x)
                if new_fit < fitness[i]:
                    population[i] = new_x
                    fitness[i] = new_fit

                    if new_fit < self.best_fitness:
                        self.best_solution = new_x.copy()
                        self.best_fitness = float(new_fit)

            # ===== ELITISMO TOP-2 (modificación 3) =====
            # Preservar los n_elites mejores reemplazando los peores
            sorted_idx = np.argsort(fitness)
            elites_idx = sorted_idx[:self.n_elites]
            worst_idx = sorted_idx[-self.n_elites:]
            for k, w in enumerate(worst_idx):
                # Solo reemplazar si la élite es mejor (siempre cierto post-sort)
                population[w] = population[elites_idx[k]].copy()
                fitness[w] = fitness[elites_idx[k]]

            self.history.append(self.best_fitness)

            if verbose and ((t + 1) % 5 == 0 or t == self.epoch - 1):
                print(f"  Época {t+1:2d}/{self.epoch}: "
                      f"a={a:.3f}  mejor fitness = {self.best_fitness:.6f}")

        return self.best_solution, self.best_fitness, self.history
