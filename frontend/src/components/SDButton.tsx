import { Button, ButtonProps } from '@chakra-ui/react';

interface Props extends ButtonProps {
    label: string;
}

const SDButton = (props: Props) => {
    const { label, size = 'sm', ...rest } = props;
    return (
        <Button size={size} {...rest}>
            {label}
        </Button>
    );
};

export default SDButton;
