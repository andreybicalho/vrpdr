import torch


def train_batch(input_tensor, target_tensor, model, optimizer,
                criterion, teacher_forcing_ratio, max_len,
                tokenizer):
    model.train()

    decoder_output = model(input_tensor, target_tensor, teacher_forcing_ratio)

    loss = 0

    optimizer.zero_grad()

    for i in range(decoder_output.size(1)):
        loss += criterion(decoder_output[:, i, :].squeeze(), target_tensor[:, i + 1])

    loss.backward()
    optimizer.step()

    target_tensor = target_tensor.cpu()
    decoder_output = decoder_output.cpu()

    prediction = torch.zeros_like(target_tensor)
    prediction[:, 0] = tokenizer.SOS_token
    for i in range(decoder_output.size(1)):
        prediction[:, i + 1] = decoder_output[:, i, :].squeeze().argmax(1)

    n_right = 0
    n_right_sentence = 0

    for i in range(prediction.size(0)):
        eq = prediction[i, 1:] == target_tensor[i, 1:]
        n_right += eq.sum().item()
        n_right_sentence += eq.all().item()

    return loss.item() / len(decoder_output), \
           n_right / prediction.size(0) / prediction.size(1), \
           n_right_sentence / prediction.size(0)


def predict_batch(input_tensor, model):
    model.eval()
    decoder_output = model(input_tensor)

    return decoder_output


def eval_batch(input_tensor, target_tensor, model, criterion, max_len, tokenizer):
    loss = 0

    decoder_output = predict_batch(input_tensor, model)

    for i in range(decoder_output.size(1)):
        loss += criterion(decoder_output[:, i, :].squeeze(), target_tensor[:, i + 1])

    target_tensor = target_tensor.cpu()
    decoder_output = decoder_output.cpu()

    prediction = torch.zeros_like(target_tensor)
    prediction[:, 0] = tokenizer.SOS_token

    for i in range(decoder_output.size(1)):
        prediction[:, i + 1] = decoder_output[:, i, :].squeeze().argmax(1)

    n_right = 0
    n_right_sentence = 0

    for i in range(prediction.size(0)):
        eq = prediction[i, 1:] == target_tensor[i, 1:]
        n_right += eq.sum().item()
        n_right_sentence += eq.all().item()

    return loss.item() / len(decoder_output), \
           n_right / prediction.size(0) / prediction.size(1), \
           n_right_sentence / prediction.size(0)