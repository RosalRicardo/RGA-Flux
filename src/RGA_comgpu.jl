using MLDatasets:MNIST
using Flux.Data:DataLoader
using Flux
using CUDA
using Zygote
using UnicodePlots

lr_g = 2e-4             # Taxa de aprendizadgem da rede geradora
lr_d = 2e-4             # Taxa de aprendizadgem da rede discriminadora
batch_size = 8          # tamanho do batch
num_epochs = 30       # Numero de Epocas de treino
output_period = 10     # duracao do periodo dos graficos de treino da rede geradora
n_features = 28 * 28    # Número de pixels em cada amostra do conjunto de dados MNIST
latent_dim = 100        # Dimensão do espaço latente
opt_dscr = ADAM(lr_d)   # Otimizador para o discriminador
opt_gen = ADAM(lr_g)    # Otimizador para o gerador

# Carrega o dataset
train_x, _ = MNIST.traindata(Float32);
# essa dataset tem pixels ∈ [0:1]. Mapeando esses pixels para [-1:1]
train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |> gpu;
#  DataLoader permite acessar dados em lote e lida com embaralhamento.
train_loader = DataLoader(train_x, batchsize=batch_size, shuffle=true);

# Definição da rede discriminadora
discriminator = Chain(Dense(n_features, 1024, x -> leakyrelu(x, 0.2f0)),
                        Dropout(0.3),
                        Dense(1024, 512, x -> leakyrelu(x, 0.2f0)),
                        Dropout(0.3),
                        Dense(512, 256, x -> leakyrelu(x, 0.2f0)),
                        Dropout(0.3),
                        Dense(256, 1, sigmoid)) |> gpu;

# Definição da rede geradora
generator = Chain(Dense(latent_dim, 256, x -> leakyrelu(x, 0.2f0)),
                        Dense(256, 512, x -> leakyrelu(x, 0.2f0)),
                        Dense(512, 1024, x -> leakyrelu(x, 0.2f0)),
                        Dense(1024, n_features, tanh)) |> gpu;

#Função de treinamento do discriminador
function train_dscr!(discriminator, real_data, fake_data)
    this_batch = size(real_data)[end] # numeros de amostras no batch
    # Concatene dados reais e falsos em um grande vetor
    all_data = hcat(real_data, fake_data)

    # Vetor de destino para previsões: 1 para dados reais, 0 para dados falsos.
    all_target = [ones(eltype(real_data), 1, this_batch) zeros(eltype(fake_data), 1, this_batch)] |> gpu;

    ps = Flux.params(discriminator)
    loss, pullback = Zygote.pullback(ps) do
        preds = discriminator(all_data)
        loss = Flux.Losses.binarycrossentropy(preds, all_target)
    end
    #Para obter os gradientes, avaliamos o retrocesso com 1.0 como um gradiente inicial.
    grads = pullback(1f0)

    # Atualize os parâmetros do discriminador com os gradientes que calculamos acima.
    Flux.update!(opt_dscr, Flux.params(discriminator), grads)
    
    return loss 
end

#Função de treinamento do gerador
function train_gen!(discriminator, generator)
    # Vetor de ruido do espaço latente
    noise = randn(latent_dim, batch_size) |> gpu;

    # Defina os parâmetros e obtenha o retorno
    ps = Flux.params(generator)
    # Avalie a função de custo ao calcular o retorno. o custo vem 'de graça'
    loss, back = Zygote.pullback(ps) do
        preds = discriminator(generator(noise));
        loss = Flux.Losses.binarycrossentropy(preds, 1.) 
    end
    # Avalie o retorno com um gradiente inicial de 1,0 para obter os gradientes para os parametros do gerador
    grads = back(1.0f0)
    Flux.update!(opt_gen, Flux.params(generator), grads)
    return loss
end

#Processo de treinamento Adversarial (Coração da GAN!)
lossvec_gen = zeros(num_epochs)
lossvec_dscr = zeros(num_epochs)

for n in 1:num_epochs
    loss_sum_gen = 0.0f0
    loss_sum_dscr = 0.0f0

    for x in train_loader
        # - Achatar as imagens de 28x28xbatchsize para 784xbatchsize
        real_data = flatten(x);

        # Treina o discriminador
        noise = randn(latent_dim, size(x)[end]) |> gpu;
        fake_data = generator(noise)
        loss_dscr = train_dscr!(discriminator, real_data, fake_data)
        loss_sum_dscr += loss_dscr

        # Treina o gerador
        loss_gen = train_gen!(discriminator, generator)
        loss_sum_gen += loss_gen
    end

    # Adicione a perda por amostra do gerador e do discriminador
    lossvec_gen[n] = loss_sum_gen / size(train_x)[end]
    lossvec_dscr[n] = loss_sum_dscr / size(train_x)[end]

    if n % output_period == 0
        @show n
        noise = randn(latent_dim, 4) |> gpu;
        fake_data = reshape(generator(noise), 28, 4*28);
        p = heatmap(fake_data, colormap=:inferno)
        print(p)
    end
end 

