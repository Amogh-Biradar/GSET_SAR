from triangulateSim import triangulateSim

def local_loss(true_tdoas, predicted_tdoas):
        mic_positions = [[0, 0], [0.05, 0], [0.025, 0.0433]]
        x_true, y_true = triangulateSim(true_tdoas, mic_positions)
        x_pred, y_pred = triangulateSim(predicted_tdoas, mic_positions)

        return ((x_true - x_pred) ** 2 + (y_true - y_pred) ** 2)

if __name__ == "__main__":
        # Example usage
        true_tdoas = [0.001, 0.002]
        predicted_tdoas = [1, 2]
        
        loss = local_loss(true_tdoas, predicted_tdoas)
        print(f"Local loss: {loss}")