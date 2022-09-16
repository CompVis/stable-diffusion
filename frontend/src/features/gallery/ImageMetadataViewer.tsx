import {
    Center,
    Flex,
    IconButton,
    Link,
    List,
    ListItem,
    Text,
} from '@chakra-ui/react';
import { FaPlus } from 'react-icons/fa';
import { PARAMETERS } from '../../app/constants';
import { useAppDispatch } from '../../app/hooks';
import SDButton from '../../components/SDButton';
import { setAllParameters, setParameter } from '../sd/sdSlice';
import { SDImage, SDMetadata } from './gallerySlice';

type Props = {
    image: SDImage;
};

const ImageMetadataViewer = ({ image }: Props) => {
    const dispatch = useAppDispatch();

    const keys = Object.keys(PARAMETERS);

    const metadata: Array<{
        label: string;
        key: string;
        value: string | number | boolean;
    }> = [];

    keys.forEach((key) => {
        const value = image.metadata[key as keyof SDMetadata];
        if (value !== undefined) {
            metadata.push({ label: PARAMETERS[key], key, value });
        }
    });

    return (
        <Flex gap={2} direction={'column'} overflowY={'scroll'} width={'100%'}>
            <SDButton
                label='Use all parameters'
                colorScheme={'gray'}
                padding={2}
                isDisabled={metadata.length === 0}
                onClick={() => dispatch(setAllParameters(image.metadata))}
            />
            <Flex gap={2}>
                <Text fontWeight={'semibold'}>File:</Text>
                <Link href={image.url} isExternal>
                    <Text>{image.url}</Text>
                </Link>
            </Flex>
            {metadata.length ? (
                <>
                    <List>
                        {metadata.map((parameter, i) => {
                            const { label, key, value } = parameter;
                            return (
                                <ListItem key={i} pb={1}>
                                    <Flex gap={2}>
                                        <IconButton
                                            aria-label='Use this parameter'
                                            icon={<FaPlus />}
                                            size={'xs'}
                                            onClick={() =>
                                                dispatch(
                                                    setParameter({
                                                        key,
                                                        value,
                                                    })
                                                )
                                            }
                                        />
                                        <Text fontWeight={'semibold'}>
                                            {label}:
                                        </Text>

                                        {value === undefined ||
                                        value === null ||
                                        value === '' ||
                                        value === 0 ? (
                                            <Text
                                                maxHeight={100}
                                                fontStyle={'italic'}
                                            >
                                                None
                                            </Text>
                                        ) : (
                                            <Text
                                                maxHeight={100}
                                                overflowY={'scroll'}
                                            >
                                                {value.toString()}
                                            </Text>
                                        )}
                                    </Flex>
                                </ListItem>
                            );
                        })}
                    </List>
                    <Flex gap={2}>
                        <Text fontWeight={'semibold'}>Raw:</Text>
                        <Text
                            maxHeight={100}
                            overflowY={'scroll'}
                            wordBreak={'break-all'}
                        >
                            {JSON.stringify(image.metadata)}
                        </Text>
                    </Flex>
                </>
            ) : (
                <Center width={'100%'} pt={10}>
                    <Text fontSize={'lg'} fontWeight='semibold'>
                        No metadata available
                    </Text>
                </Center>
            )}
        </Flex>
    );
};

export default ImageMetadataViewer;
