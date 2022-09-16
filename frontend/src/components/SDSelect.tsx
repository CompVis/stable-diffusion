import {
    Flex,
    FormControl,
    FormLabel,
    Select,
    SelectProps,
    Text,
} from '@chakra-ui/react';

interface Props extends SelectProps {
    label: string;
    validValues:
        | Array<number | string>
        | Array<{ key: string; value: string | number }>;
}

const SDSelect = (props: Props) => {
    const {
        label,
        isDisabled,
        validValues,
        size = 'sm',
        fontSize = 'md',
        marginBottom = 1,
        whiteSpace = 'nowrap',
        ...rest
    } = props;
    return (
        <FormControl isDisabled={isDisabled}>
            <Flex justifyContent={'space-between'} alignItems={'center'}>
                <FormLabel
                    marginBottom={marginBottom}
                >
                    <Text fontSize={fontSize} whiteSpace={whiteSpace}>
                        {label}
                    </Text>
                </FormLabel>
                <Select fontSize={fontSize} size={size} {...rest}>
                    {validValues.map((opt) => {
                        return typeof opt === 'string' ||
                            typeof opt === 'number' ? (
                            <option key={opt} value={opt}>
                                {opt}
                            </option>
                        ) : (
                            <option key={opt.value} value={opt.value}>
                                {opt.key}
                            </option>
                        );
                    })}
                </Select>
            </Flex>
        </FormControl>
    );
};

export default SDSelect;
