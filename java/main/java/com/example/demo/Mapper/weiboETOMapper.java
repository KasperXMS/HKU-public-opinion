package com.example.demo.Mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.example.demo.Entity.tweetDto;
import com.example.demo.Entity.weiboETO;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

@Mapper
public interface weiboETOMapper extends BaseMapper<weiboETO> {
    @Select("${findtext}")
    List<weiboETO> FindText(@Param("findtext")String sql);
}
