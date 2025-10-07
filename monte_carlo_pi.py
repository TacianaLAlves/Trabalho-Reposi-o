import random
import math
import time
from typing import Tuple

def monte_carlo_pi_python(n_points: int) -> Tuple[float, float, int]:
    """
    Calcula π usando o método de Monte Carlo (Python puro)
    
    Args:
        n_points: Número de pontos a gerar
        
    Returns:
        Tuple: (pi_approximado, erro_absoluto, pontos_dentro)
    """
    pontos_dentro = 0
    
    for _ in range(n_points):
        # Gerar ponto aleatório no quadrado [0,1] x [0,1]
        x = random.random()
        y = random.random()
        
        # Verificar se está dentro do quarto de círculo (x² + y² ≤ 1)
        if x*x + y*y <= 1.0:
            pontos_dentro += 1
    
    # Calcular π aproximado
    pi_approx = 4.0 * pontos_dentro / n_points
    erro = abs(pi_approx - math.pi)
    
    return pi_approx, erro, pontos_dentro

def monte_carlo_pi_python_otimizado(n_points: int) -> Tuple[float, float, int]:
    """
    Versão otimizada usando compreensão de lista
    """
    pontos_dentro = sum(1 for _ in range(n_points) 
                       if random.random()**2 + random.random()**2 <= 1.0)
    
    pi_approx = 4.0 * pontos_dentro / n_points
    erro = abs(pi_approx - math.pi)
    
    return pi_approx, erro, pontos_dentro

# Teste e análise de performance
def testar_monte_carlo_python():
    """Testa a implementação Python com diferentes números de pontos"""
    
    print("🧪 TESTE MÉTODO MONTE CARLO - PYTHON PURO")
    print("=" * 60)
    
    tamanhos_teste = [1000, 10000, 100000, 1000000]
    
    for n in tamanhos_teste:
        inicio = time.time()
        pi_approx, erro, pontos_dentro = monte_carlo_pi_python(n)
        tempo = time.time() - inicio
        
        print(f"Pontos: {n:>8,}")
        print(f"  π aproximado: {pi_approx:.8f}")
        print(f"  π real:       {math.pi:.8f}")
        print(f"  Erro:         {erro:.8f}")
        print(f"  Pontos dentro: {pontos_dentro}/{n} ({pontos_dentro/n*100:.2f}%)")
        print(f"  Tempo:        {tempo:.4f} segundos")
        print("-" * 40)

# Executar teste
testar_monte_carlo_python()

from numba import jit
import numpy as np

@jit(nopython=True, nogil=True, cache=True)
def monte_carlo_pi_numba(n_points: int) -> Tuple[float, float, int]:
    """
    Calcula π usando o método de Monte Carlo (acelerado com Numba)
    
    Args:
        n_points: Número de pontos a gerar
        
    Returns:
        Tuple: (pi_approximado, erro_absoluto, pontos_dentro)
    """
    pontos_dentro = 0
    rng = np.random.default_rng()
    
    for _ in range(n_points):
        # Gerar ponto aleatório usando numpy (mais rápido com Numba)
        x = rng.random()
        y = rng.random()
        
        # Verificar se está dentro do quarto de círculo
        if x*x + y*y <= 1.0:
            pontos_dentro += 1
    
    # Calcular π aproximado
    pi_approx = 4.0 * pontos_dentro / n_points
    erro = abs(pi_approx - np.pi)
    
    return pi_approx, erro, pontos_dentro

@jit(nopython=True, nogil=True, cache=True)
def monte_carlo_pi_numba_otimizado(n_points: int) -> Tuple[float, float, int]:
    """
    Versão ainda mais otimizada usando operações vetorizadas
    """
    rng = np.random.default_rng()
    
    # Gerar todos os pontos de uma vez (mais eficiente)
    pontos = rng.random((n_points, 2))
    distancias = pontos[:, 0]**2 + pontos[:, 1]**2
    pontos_dentro = np.sum(distancias <= 1.0)
    
    pi_approx = 4.0 * pontos_dentro / n_points
    erro = abs(pi_approx - np.pi)
    
    return pi_approx, erro, pontos_dentro

def testar_monte_carlo_numba():
    """Testa a implementação Numba com diferentes números de pontos"""
    
    print("🚀 TESTE MÉTODO MONTE CARLO - NUMBA ACELERADO")
    print("=" * 60)
    
    tamanhos_teste = [1000, 10000, 100000, 1000000, 5000000, 10000000]
    
    for n in tamanhos_teste:
        inicio = time.time()
        pi_approx, erro, pontos_dentro = monte_carlo_pi_numba_otimizado(n)
        tempo = time.time() - inicio
        
        print(f"Pontos: {n:>10,}")
        print(f"  π aproximado: {pi_approx:.8f}")
        print(f"  π real:       {np.pi:.8f}")
        print(f"  Erro:         {erro:.8f}")
        print(f"  Pontos dentro: {pontos_dentro:>8,}/{n} ({pontos_dentro/n*100:.2f}%)")
        print(f"  Tempo:        {tempo:.4f} segundos")
        print("-" * 50)

# Executar teste Numba
testar_monte_carlo_numba()

def comparar_performances():
    """Compara performance entre Python puro e Numba"""
    
    print("⚡ COMPARAÇÃO DE PERFORMANCE: PYTHON PURO vs NUMBA")
    print("=" * 70)
    
    tamanhos = [10000, 100000, 1000000]
    
    resultados = []
    
    for n in tamanhos:
        # Teste Python puro
        inicio_py = time.time()
        pi_py, erro_py, dentro_py = monte_carlo_pi_python(n)
        tempo_py = time.time() - inicio_py
        
        # Teste Numba (primeira execução inclui compilação)
        inicio_nb = time.time()
        pi_nb, erro_nb, dentro_nb = monte_carlo_pi_numba(n)
        tempo_nb = time.time() - inicio_nb
        
        # Teste Numba otimizado
        inicio_nb_opt = time.time()
        pi_nb_opt, erro_nb_opt, dentro_nb_opt = monte_carlo_pi_numba_otimizado(n)
        tempo_nb_opt = time.time() - inicio_nb_opt
        
        speedup = tempo_py / tempo_nb_opt if tempo_nb_opt > 0 else 0
        
        resultados.append({
            'n': n,
            'tempo_python': tempo_py,
            'tempo_numba': tempo_nb,
            'tempo_numba_opt': tempo_nb_opt,
            'speedup': speedup,
            'pi_python': pi_py,
            'pi_numba': pi_nb_opt
        })
        
        print(f"\n🔍 {n:,} PONTOS:")
        print(f"  Python puro:    {tempo_py:.6f}s - π = {pi_py:.6f}")
        print(f"  Numba:          {tempo_nb:.6f}s - π = {pi_nb:.6f}")
        print(f"  Numba otimizado: {tempo_nb_opt:.6f}s - π = {pi_nb_opt:.6f}")
        print(f"  ⚡ SPEEDUP:     {speedup:.1f}x mais rápido")
    
    return resultados

# Executar comparação
resultados = comparar_performances()

import matplotlib.pyplot as plt
import numpy as np

def visualizar_monte_carlo(n_points=10000):
    """
    Visualiza graficamente o método de Monte Carlo
    """
    # Gerar pontos
    pontos = np.random.random((n_points, 2))
    distancias = pontos[:, 0]**2 + pontos[:, 1]**2
    dentro = distancias <= 1.0
    fora = ~dentro
    
    # Calcular π
    pi_approx = 4.0 * np.sum(dentro) / n_points
    
    # Criar visualização
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfico 1: Pontos
    ax1.scatter(pontos[dentro, 0], pontos[dentro, 1], 
               color='blue', alpha=0.6, s=1, label=f'Dentro ({np.sum(dentro)})')
    ax1.scatter(pontos[fora, 0], pontos[fora, 1], 
               color='red', alpha=0.6, s=1, label=f'Fora ({np.sum(fora)})')
    
    # Adicionar quarto de círculo
    theta = np.linspace(0, np.pi/2, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    ax1.plot(x_circle, y_circle, 'k-', linewidth=2, label='Quarto de círculo')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Método de Monte Carlo para π\n'
                 f'π ≈ 4 × {np.sum(dentro)}/{n_points} = {pi_approx:.6f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfico 2: Convergência
    pontos_parciais = np.arange(100, n_points, 100)
    pi_parcial = [4.0 * np.sum(distancias[:n] <= 1.0) / n for n in pontos_parciais]
    
    ax2.plot(pontos_parciais, pi_parcial, 'b-', alpha=0.7, label='π aproximado')
    ax2.axhline(y=np.pi, color='r', linestyle='--', label=f'π real = {np.pi:.8f}')
    ax2.set_xlabel('Número de Pontos')
    ax2.set_ylabel('Valor de π')
    ax2.set_title('Convergência do Método de Monte Carlo')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(3.0, 3.3)
    
    plt.tight_layout()
    plt.show()
    
    return pi_approx

# Gerar visualização
pi_visual = visualizar_monte_carlo(5000)
print(f"🎯 π aproximado na visualização: {pi_visual:.8f}")


def analise_estatistica_monte_carlo(n_simulacoes=100, n_pontos=10000):
    """
    Análise estatística do método de Monte Carlo
    """
    print("📊 ANÁLISE ESTATÍSTICA DO MÉTODO MONTE CARLO")
    print("=" * 50)
    
    resultados = []
    
    for i in range(n_simulacoes):
        pi_approx, erro, _ = monte_carlo_pi_numba_otimizado(n_pontos)
        resultados.append(pi_approx)
    
    resultados = np.array(resultados)
    
    print(f"Simulações: {n_simulacoes}")
    print(f"Pontos por simulação: {n_pontos:,}")
    print(f"π médio: {np.mean(resultados):.8f}")
    print(f"π real:  {np.pi:.8f}")
    print(f"Viés:    {np.mean(resultados) - np.pi:+.8f}")
    print(f"Desvio padrão: {np.std(resultados):.8f}")
    print(f"Erro médio absoluto: {np.mean(np.abs(resultados - np.pi)):.8f}")
    print(f"Intervalo 95%: [{np.percentile(resultados, 2.5):.6f}, "
          f"{np.percentile(resultados, 97.5):.6f}]")
    
    # Histograma
    plt.figure(figsize=(10, 6))
    plt.hist(resultados, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.pi, color='red', linestyle='--', linewidth=2, 
                label=f'π real = {np.pi:.8f}')
    plt.axvline(np.mean(resultados), color='green', linestyle='--', linewidth=2,
                label=f'π médio = {np.mean(resultados):.8f}')
    plt.xlabel('Valor de π Aproximado')
    plt.ylabel('Frequência')
    plt.title(f'Distribuição de {n_simulacoes} Simulações de Monte Carlo\n'
              f'({n_pontos:,} pontos cada)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return resultados

# Executar análise estatística
resultados_stats = analise_estatistica_monte_carlo(50, 5000)
