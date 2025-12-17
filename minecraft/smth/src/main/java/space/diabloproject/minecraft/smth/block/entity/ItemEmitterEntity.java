package space.diabloproject.minecraft.smth.block.entity;

import net.minecraft.core.BlockPos;
import net.minecraft.core.NonNullList;
import net.minecraft.network.chat.Component;
import net.minecraft.world.MenuProvider;
import net.minecraft.world.entity.player.Inventory;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.inventory.AbstractContainerMenu;
import net.minecraft.world.inventory.DispenserMenu;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.block.entity.BaseContainerBlockEntity;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.entity.BlockEntityType;
import net.minecraft.world.level.block.state.BlockState;
import org.jetbrains.annotations.Nullable;
import space.diabloproject.minecraft.smth.ModBlocks;

public class ItemEmitterEntity extends BaseContainerBlockEntity {
    private NonNullList<ItemStack> items;
    public ItemEmitterEntity(BlockPos blockPos, BlockState blockState) {
        super(ModBlocks.ITEM_EMITTER_ENTITY, blockPos, blockState);
        this.items = NonNullList.withSize(9, ItemStack.EMPTY);
    }

    @Override
    protected Component getDefaultName() {
        return Component.literal("Item Emitter");
    }

    @Override
    protected NonNullList<ItemStack> getItems() {
        return this.items;
    }

    @Override
    protected void setItems(NonNullList<ItemStack> items) {
        this.items = items;
    }

    @Override
    protected AbstractContainerMenu createMenu(int i, Inventory inventory) {
        return new DispenserMenu(i, inventory, this);
    }

    @Override
    public int getContainerSize() {
        return 9;
    }
}
