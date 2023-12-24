import torch
import torch.nn as nn
from generator_model import Generator
from discriminator_model import Discriminator
import sys
from torch.utils.data import DataLoader
from utils import save_checkpoint, load_checkpoint
from dataset import LandscapePaintDataset
import torch.optim as optim
from tqdm import tqdm
import config
from torchvision.utils import save_image
import pickle


def train_fn(gen_paint, gen_landscape, disc_paint, disc_landscape,
          train_loader, opt_disc, opt_gen, L1_Loss, mse, g_scaler, d_scaler):

    loop = tqdm(train_loader, leave=True)

    for index, (landscape, paint) in enumerate(loop):
        landscape = landscape.to(config.DEVICE)
        paint = paint.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():

            # Fake landscape
            fake_landscape = gen_landscape(paint)

            DLandscape_real = disc_landscape(landscape)
            DLandscape_fake = disc_landscape(fake_landscape)

            DLandscape_real_loss = mse(DLandscape_real, torch.ones_like(DLandscape_real))
            DLandscape_fake_loss = mse(DLandscape_fake, torch.zeros_like(DLandscape_fake))

            DLandscape_loss = DLandscape_real_loss + DLandscape_fake_loss

            # Fake paint
            fake_paint = gen_paint(landscape)

            DPaint_real = disc_paint(paint)
            DPaint_fake = disc_paint(fake_paint)

            DPaint_real_loss = mse(DPaint_real, torch.ones_like(DPaint_real))
            DPaint_fake_loss = mse(DPaint_fake, torch.zeros_like(DPaint_fake))

            DPaint_loss = DPaint_real_loss + DPaint_fake_loss

            # Discriminator loss
            disc_loss = 0.5 * (DPaint_loss + DLandscape_loss)

        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward(retain_graph=True)
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generator
        with torch.cuda.amp.autocast():
            DPaint_fake = disc_paint(fake_paint)
            DLandscape_fake = disc_landscape(fake_landscape)

            lossG_paint = mse(DPaint_fake, torch.ones_like(DPaint_fake))
            lossG_landscape = mse(DLandscape_fake, torch.ones_like(DLandscape_fake))
            lossG = 0.5 * (lossG_landscape + lossG_paint)

            # Cycle consistency loss
            cycle_paint = gen_paint(fake_landscape)
            cycle_landscape = gen_landscape(fake_paint)

            lossC_paint = L1_Loss(cycle_paint, paint)
            lossC_landscape = L1_Loss(cycle_landscape, landscape)
            lossC = 0.5 * (lossC_landscape + lossC_paint)

            # Identity loss
            identity_paint = gen_paint(paint)
            identity_landscape = gen_landscape(landscape)

            lossI_paint = L1_Loss(identity_paint, paint)
            lossI_landscape = L1_Loss(identity_landscape, landscape)
            lossI = 0.5 * (lossI_paint + lossI_landscape)

            # Overall generator loss:
            gen_loss = lossG + config.LAMBDA_CYCLE * lossC + config.LAMBDA_IDENTITY * lossI

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()


def main():
    disc_landscape = Discriminator(in_channels=3).to(config.DEVICE)
    disc_paint = Discriminator(in_channels=3).to(config.DEVICE)
    gen_landscape = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_paint = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    L1_Loss = nn.L1Loss()  # For cycle consistency loss and identity loss
    mse = nn.MSELoss()  # For standard GAN Loss

    opt_disc = optim.Adam(list(disc_landscape.parameters()) + list(disc_paint.parameters()),
                          lr=config.LEARNING_RATE,
                          betas=[0.5, 0.999]
    )

    opt_gen = optim.Adam(list(gen_paint.parameters()) + list(gen_landscape.parameters()),
                         lr=config.LEARNING_RATE,
                         betas=[0.5, 0.999]
    )

    if config.LOAD_MODEL:

        load_checkpoint(
            config.CHECKPOINT_GEN_P, gen_paint, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_L, gen_landscape, opt_gen, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_L, disc_landscape, opt_disc, config.LEARNING_RATE
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_P, disc_paint, opt_disc, config.LEARNING_RATE
        )

    train_dataset = LandscapePaintDataset(landscape_dir=config.TRAIN_DIR + "/landscape",
                                          paint_dir=config.TRAIN_DIR + "/painting",
                                          transforms=config.transforms
                                         )

    val_dataset = LandscapePaintDataset(landscape_dir=config.VAL_DIR + "/landscape",
                                        paint_dir=config.VAL_DIR + "/painting",
                                        transforms=config.transforms
                                       )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True

    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(gen_paint, gen_landscape, disc_paint, disc_landscape,
                 train_loader, opt_disc, opt_gen, L1_Loss, mse, g_scaler, d_scaler)

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen_landscape, opt_gen, config.CHECKPOINT_GEN_L)
            save_checkpoint(gen_paint, opt_gen, config.CHECKPOINT_GEN_P)
            save_checkpoint(disc_paint, opt_disc, config.CHECKPOINT_CRITIC_P)
            save_checkpoint(disc_landscape, opt_disc, config.CHECKPOINT_CRITIC_L)

            for idx, (landscape, paint) in enumerate(val_loader):
                fake_paint = gen_paint(landscape)
                fake_landscape = gen_landscape(paint)
                break

                save_image(fake_paint * 0.5 + 0.5, f"saved_images/paint_{epoch}.png")
                save_image(fake_landscape * 0.5 + 0.5, f"saved_images/landscape_{epoch}.png")

    gen_paint_path = "paint_generator.pkl"
    gen_landscape_path = "landscape_generator.pkl"

    with open(gen_paint_path, 'wb') as f:
        pickle.dump(gen_paint, f)

    with open(gen_landscape_path, 'wb') as f2:
        pickle.dump(gen_landscape, f2)


if __name__ == "__main__":
    main()