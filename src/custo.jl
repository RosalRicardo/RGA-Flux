module funcoes_custo

using StatsBase, Distributions, Random, BenchmarkTools

function divergencia_kl(p,q)
    """
    Retorna a divergencia KL entre duas distribuições.
            Parameters:
                    p (array): distribuição de probabilidade
                    q (array): distribuição de probabilidade
            Returns:
                    divergencia kl (scalar): retorna valor que escalar que pode ser interpretado como a divergencia entre as distribuições
    """
    return StatsBase.kldivergence(p,q) 
end

function distancia_wasserstein(p,q)
    """
    Retorna a distancia de Wasserstein entre duas distribuições.
            Parameters:
                    p (array): distribuição de probabilidade
                    q (array): distribuição de probabilidade
            Returns:
                    distancia de Wasserstein (scalar): retorna valor que escalar que pode ser interpretado como a divergencia entre as distribuições
    """

    return  abs(cdf(p)-cdf(p))
end

end #fim do modulo