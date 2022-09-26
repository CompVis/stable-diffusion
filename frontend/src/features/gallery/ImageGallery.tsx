import { Button, Center, Flex, Text } from '@chakra-ui/react';
import { requestImages } from '../../app/socketio/actions';
import { RootState, useAppDispatch } from '../../app/store';
import { useAppSelector } from '../../app/store';
import HoverableImage from './HoverableImage';

/**
 * Simple image gallery.
 */
const ImageGallery = () => {
  const { images, currentImageUuid } = useAppSelector(
    (state: RootState) => state.gallery
  );
  const dispatch = useAppDispatch();
  /**
   * I don't like that this needs to rerender whenever the current image is changed.
   * What if we have a large number of images? I suppose pagination (planned) will
   * mitigate this issue.
   *
   * TODO: Refactor if performance complaints, or after migrating to new API which supports pagination.
   */

  const handleClickLoadMore = () => {
    dispatch(requestImages());
  };

  return images.length ? (
    <Flex direction={'column'} gap={2} pb={2}>
      <Flex gap={2} wrap="wrap">
        {images.map((image) => {
          const { uuid } = image;
          const isSelected = currentImageUuid === uuid;
          return (
            <HoverableImage key={uuid} image={image} isSelected={isSelected} />
          );
        })}
      </Flex>
      <Button onClick={handleClickLoadMore}>Load more...</Button>
    </Flex>
  ) : (
    <Center height={'100%'} position={'relative'}>
      <Text size={'xl'}>No images in gallery</Text>
    </Center>
  );
};

export default ImageGallery;
