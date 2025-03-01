def testAttentionSingleLayer():
    from Emperor.base.utils import Trainer
    from Emperor.base.datasets import FashionMNIST
    from Emperor.experiments import AttentionSingleLayerModel
    from Emperor.config import ModelConfig, ParameterGeneratorOptions

    cfg = ModelConfig(
        batchSize=16,
        inputDim=16,
        hiddenDim=64,
        outputDim=10,
        depthDim=32,
        embeddingDim=16,
        qkvHiddenDim=64,
        attentionOutputDim=10,
        numExperts=6,
        headDim=32,
        parameterGeneartorType=ParameterGeneratorOptions.matrix_choice_sparse,
    )

    data = FashionMNIST(
        batch_size=cfg.batchSize, testDatasetFalg=True, testDatasetNumSamples=32
    )

    model = AttentionSingleLayerModel(learningRate=0.1, cfg=cfg)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data, printLossFlag=True)


def testTransformerEncoderLayerBaseSingleLayerModel():
    from Emperor.base.utils import Trainer
    from Emperor.base.datasets import FashionMNIST
    from Emperor.experiments import TransformerEncoderLayerBaseSingleLayerModel
    from Emperor.config import ModelConfig, ParameterGeneratorOptions

    cfg = ModelConfig(
        batchSize=128,
        inputDim=16,
        depthDim=64,
        embeddingDim=16,
        qkvHiddenDim=64,
        numExperts=16,
        headDim=32,
        topK=2,
        parameterGeneartorType=ParameterGeneratorOptions.matrix_choice_sparse,
    )

    data = FashionMNIST(
        batch_size=cfg.batchSize, testDatasetFalg=True, testDatasetNumSamples=1024
    )

    model = TransformerEncoderLayerBaseSingleLayerModel(learningRate=0.1, cfg=cfg)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data, printLossFlag=True)


def testTransformerDecoderLayerBaseSingleLayerModel():
    from Emperor.base.utils import Trainer
    from Emperor.base.datasets import FashionMNIST
    from Emperor.experiments import TransformerDecoderLayerBaseSingleLayerModel
    from Emperor.config import ModelConfig, ParameterGeneratorOptions

    cfg = ModelConfig(
        batchSize=16,
        inputDim=16,
        depthDim=64,
        embeddingDim=16,
        qkvHiddenDim=64,
        numExperts=16,
        headDim=32,
        topK=2,
        parameterGeneartorType=ParameterGeneratorOptions.matrix_choice_sparse,
    )

    data = FashionMNIST(
        batch_size=cfg.batchSize, testDatasetFalg=True, testDatasetNumSamples=16
    )

    model = TransformerDecoderLayerBaseSingleLayerModel(learningRate=0.1, cfg=cfg)
    trainer = Trainer(max_epochs=100)
    trainer.fit(model, data, printLossFlag=True)


if __name__ == "__main__":
    # testAttentionSingleLayer()
    # testTransformerEncoderLayerBaseSingleLayerModel()

    testTransformerDecoderLayerBaseSingleLayerModel()
